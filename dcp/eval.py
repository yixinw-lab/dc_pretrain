import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dcp.runtime import get_dist_info


def _model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    model_device = _model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model_device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model_device)
    it = iter(batches)
    for _ in range(steps):
        x, y, _ = next(it)
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


def _local_eval_token_budget(total_tokens: int) -> int:
    _, rank, _, world_size = get_dist_info()
    base = total_tokens // world_size
    remainder = total_tokens % world_size
    return base + (1 if rank < remainder else 0)


def _iter_eval_token_budget(batches, total_tokens: int):
    remaining = _local_eval_token_budget(total_tokens)
    if remaining <= 0:
        return
    batch_iter = iter(batches)
    while remaining > 0:
        x, y, _ = next(batch_iter)
        B, T = x.shape
        batch_tokens = B * T
        if remaining >= batch_tokens:
            yield x, y
            remaining -= batch_tokens
            continue
        full_rows, tail_tokens = divmod(remaining, T)
        if full_rows > 0:
            yield x[:full_rows], y[:full_rows]
        if tail_tokens > 0:
            yield x[full_rows:full_rows + 1, :tail_tokens], y[full_rows:full_rows + 1, :tail_tokens]
        break


def _bucket_eval_length(required_length, T, crop_bucket):
    if required_length >= T:
        return T
    if crop_bucket <= 1:
        return required_length
    bucketed = ((required_length + crop_bucket - 1) // crop_bucket) * crop_bucket
    return min(T, max(required_length, bucketed))


def _build_bidirectional_eval_attn_mask(L, live_length, right_window_size, device):
    row = torch.arange(L, device=device).view(L, 1)
    col = torch.arange(L, device=device).view(1, L)
    causal = col <= row
    within_live = col < live_length
    future = (col > row) & (col <= torch.clamp(row + right_window_size, max=live_length - 1))
    return (causal | (future & within_live)).view(1, 1, L, L)


@torch.no_grad()
def _evaluate_bidirectional_batch(model, x, y, mask_token_id, ltr_length=16, eval_right_window_size=16, crop_bucket=256):
    B, T = x.shape
    losses = []
    counts = []
    for t in range(ltr_length, T):
        live_length = min(T, t + 1 + eval_right_window_size)
        L = _bucket_eval_length(live_length, T, crop_bucket)
        masked_x = x[:, :L].clone()
        targets = y[:, :L]
        if t + 1 < live_length:
            masked_x[:, t + 1:live_length] = mask_token_id
        attn_mask = _build_bidirectional_eval_attn_mask(
            L, live_length, eval_right_window_size, x.device
        )
        logits = model(masked_x, attn_mask=attn_mask)
        row_loss = F.cross_entropy(logits[:, t], y[:, t], ignore_index=-1, reduction='none')
        valid = y[:, t] != -1
        losses.append((row_loss * valid).sum())
        counts.append(valid.sum())
    if not losses:
        zero = torch.tensor(0.0, device=x.device)
        return zero, zero
    return torch.stack(losses).sum(), torch.stack(counts).sum()


@torch.no_grad()
def evaluate_bpb_token_budget(model, batches, total_tokens, token_bytes):
    model_device = _model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model_device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_tokens_seen = torch.tensor(0, dtype=torch.int64, device=model_device)
    for x, y in _iter_eval_token_budget(batches, total_tokens):
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens_seen += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_seen, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens_seen = total_loss.item(), total_tokens_seen.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens_seen if total_tokens_seen > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_bidirectional_loss_token_budget(model, batches, total_tokens, mask_token_id, ltr_length=16, eval_right_window_size=16, crop_bucket=256):
    model_device = _model_device(model)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_count = torch.tensor(0, dtype=torch.int64, device=model_device)
    for x, y in _iter_eval_token_budget(batches, total_tokens):
        batch_loss, batch_count = _evaluate_bidirectional_batch(
            model,
            x,
            y,
            mask_token_id,
            ltr_length=ltr_length,
            eval_right_window_size=eval_right_window_size,
            crop_bucket=crop_bucket,
        )
        total_loss += batch_loss
        total_count += batch_count
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    return total_loss.item() / max(int(total_count.item()), 1)


@torch.no_grad()
def evaluate_mdm_loss_token_budget(model, batches, total_tokens, mask_token_id, mask_eps, compute_ddc_loss_fn, num_mc=1):
    model_device = _model_device(model)
    total_weighted_loss = torch.tensor(0.0, dtype=torch.float32, device=model_device)
    total_slots = torch.tensor(0, dtype=torch.int64, device=model_device)
    for x, _ in _iter_eval_token_budget(batches, total_tokens):
        for _ in range(num_mc):
            loss, _ = compute_ddc_loss_fn(
                model,
                x,
                mask_token_id=mask_token_id,
                mask_eps=mask_eps,
                ar_mask_ratio=0.0,
            )
            total_weighted_loss += loss * (x.size(0) * x.size(1))
            total_slots += x.size(0) * x.size(1)
    if dist.is_initialized():
        dist.all_reduce(total_weighted_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_slots, op=dist.ReduceOp.SUM)
    return total_weighted_loss.item() / max(int(total_slots.item()), 1)
