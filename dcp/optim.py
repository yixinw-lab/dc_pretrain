import math

import torch
import torch.distributed as dist


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))


class DistShardedAdamW(torch.optim.Optimizer):
    """Distributed AdamW with ZeRO-2 style state sharding for large tensors."""

    def __init__(self, param_groups, max_grad_norm, return_grad_norm=False):
        super().__init__(param_groups, defaults={})
        self.max_grad_norm = float(max_grad_norm)
        self.return_grad_norm = bool(return_grad_norm)
        self._step_t = torch.tensor(0.0)
        self._lr_t = torch.tensor(0.0)
        self._beta1_t = torch.tensor(0.0)
        self._beta2_t = torch.tensor(0.0)
        self._eps_t = torch.tensor(0.0)
        self._wd_t = torch.tensor(0.0)

    def _world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    def _rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def _should_shard(self, p, world_size):
        grad = p.grad
        return (
            world_size > 1
            and grad is not None
            and grad.ndim > 0
            and p.numel() >= 1024
            and grad.shape[0] % world_size == 0
        )

    def _reduce_group(self, group, world_size):
        infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if self._should_shard(p, world_size):
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
            else:
                future = None
                if world_size > 1:
                    future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
        return dict(param_infos=infos)

    def _wait_reductions(self, reduce_infos):
        for info in reduce_infos:
            for pinfo in info["param_infos"].values():
                if pinfo["future"] is not None:
                    pinfo["future"].wait()

    def _grad_norm_device(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return p.grad.device
        return torch.device("cpu")

    def _compute_global_grad_norm(self, reduce_infos, world_size):
        total_sq = torch.zeros((), dtype=torch.float64, device=self._grad_norm_device())
        for info in reduce_infos:
            for pinfo in info["param_infos"].values():
                grad_sq = pinfo["grad_slice"].detach().float().square().sum(dtype=torch.float64)
                if pinfo["is_small"] and world_size > 1:
                    grad_sq /= world_size
                total_sq += grad_sq
        if world_size > 1:
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        return total_sq.sqrt().item()

    def _build_adamw_entries(self, group, info, rank, world_size):
        entries = []
        for p in group["params"]:
            pinfo = info["param_infos"].get(p)
            if pinfo is None:
                continue
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            entries.append((p, pinfo, p_slice, state))
        return entries

    def _step_adamw_single(self, p, pinfo, p_slice, state, group, clip_coef, gather_futures):
        grad_slice = pinfo["grad_slice"]
        if clip_coef < 1.0:
            grad_slice.mul_(clip_coef)
        state["step"] += 1
        self._step_t.fill_(state["step"])
        self._lr_t.fill_(group["lr"])
        self._beta1_t.fill_(group["betas"][0])
        self._beta2_t.fill_(group["betas"][1])
        self._eps_t.fill_(group["eps"])
        self._wd_t.fill_(group["weight_decay"])
        adamw_step_fused(
            p_slice,
            grad_slice,
            state["exp_avg"],
            state["exp_avg_sq"],
            self._step_t,
            self._lr_t,
            self._beta1_t,
            self._beta2_t,
            self._eps_t,
            self._wd_t,
        )
        if not pinfo["is_small"]:
            future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
            gather_futures.append(future)

    def _step_adamw_bucket(self, bucket_entries, group, clip_coef, gather_futures):
        if not bucket_entries:
            return
        if len(bucket_entries) == 1:
            self._step_adamw_single(*bucket_entries[0], group, clip_coef, gather_futures)
            return

        current_steps = []
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        for p, pinfo, p_slice, state in bucket_entries:
            current_steps.append(state["step"])
            params.append(p_slice)
            grads.append(pinfo["grad_slice"])
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

        if any(step != current_steps[0] for step in current_steps[1:]):
            for entry in bucket_entries:
                self._step_adamw_single(*entry, group, clip_coef, gather_futures)
            return

        step = current_steps[0] + 1
        for _, _, _, state in bucket_entries:
            state["step"] = step

        if clip_coef < 1.0:
            torch._foreach_mul_(grads, clip_coef)

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step

        torch._foreach_mul_(params, 1 - lr * wd)
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        grad_sq = torch._foreach_mul(grads, grads)
        torch._foreach_lerp_(exp_avg_sqs, grad_sq, 1 - beta2)
        denom = torch._foreach_div(exp_avg_sqs, bias2)
        denom = torch._foreach_sqrt(denom)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-(lr / bias1))

        for p, pinfo, p_slice, _ in bucket_entries:
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_futures.append(future)

    def _compute_adamw(self, group, info, rank, world_size, clip_coef, gather_futures):
        entries = self._build_adamw_entries(group, info, rank, world_size)
        buckets = {}
        for entry in entries:
            _, pinfo, p_slice, _ = entry
            key = (pinfo["is_small"], tuple(p_slice.shape), p_slice.dtype)
            buckets.setdefault(key, []).append(entry)
        for bucket_entries in buckets.values():
            self._step_adamw_bucket(bucket_entries, group, clip_coef, gather_futures)

    @torch.no_grad()
    def step(self):
        world_size = self._world_size()
        rank = self._rank()
        reduce_infos = [self._reduce_group(group, world_size) for group in self.param_groups]
        self._wait_reductions(reduce_infos)
        grad_norm = self._compute_global_grad_norm(reduce_infos, world_size)
        clip_coef = 1.0
        if self.max_grad_norm > 0.0 and math.isfinite(grad_norm) and grad_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
        gather_futures = []
        for group, info in zip(self.param_groups, reduce_infos):
            self._compute_adamw(group, info, rank, world_size, clip_coef, gather_futures)
        for future in gather_futures:
            future.wait()
        if self.return_grad_norm:
            return {"grad_norm": grad_norm}
        return None

