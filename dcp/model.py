import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcp.runtime import print0


def _compute_mlp_hidden_dim(n_embd: int) -> int:
    return 256 * ((8 * n_embd // 3 + 255) // 256)


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_fa3()


def fa3_available():
    return _fa3 is not None


def _merge_with_causal_window(attn_mask, q, k, window_size):
    Tq, Tk = q.size(2), k.size(2)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    window = window_size[0]
    if attn_mask is None:
        base = col_idx <= row_idx
        if window >= 0 and window < Tk:
            base = base & ((row_idx - col_idx) <= window)
        return base.view(1, 1, Tq, Tk)
    if attn_mask.dtype != torch.bool:
        raise TypeError("custom attention mask must be boolean")
    causal = (col_idx <= row_idx).view(1, 1, Tq, Tk)
    left = attn_mask & causal
    if window >= 0 and window < Tk:
        left_window = ((row_idx - col_idx) <= window).view(1, 1, Tq, Tk)
        left = left & left_window
    extra_right = attn_mask & (~causal)
    return left | extra_right


def _sdpa_attention(q, k, v, window_size, enable_gqa, attn_mask=None, causal=False):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if attn_mask is None and (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)
    if attn_mask is None and Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    merged_mask = _merge_with_causal_window(attn_mask, q, k, window_size)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=merged_mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), attn_mask=None):
    """Flash Attention for training. q,k,v: (B, T, H, D)."""
    if attn_mask is None and _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa, attn_mask=attn_mask, causal=causal)
    return y.transpose(1, 2)


flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    window_pattern: str = "SSSL"
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, (x.size(-1),), weight=weight, eps=self.eps)


def _init_trunc_normal_(tensor, std):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _linear_fan_in_std(linear):
    return linear.in_features ** -0.5


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos_sin, window_size, attn_mask=None, causal=True):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=causal, window_size=window_size, attn_mask=attn_mask)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = _compute_mlp_hidden_dim(config.n_embd)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin, window_size, attn_mask=None, causal=True):
        x = x + self.attn(self.attn_norm(x), cos_sin, window_size, attn_mask=attn_mask, causal=causal)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64, use_full_window_sizes=False, manual_ignore_index_loss=False):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        if use_full_window_sizes:
            self.full_window_sizes = [(-1, -1)] * config.n_layer
        self.manual_ignore_index_loss = manual_ignore_index_loss
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def apply_runtime_precision_fixup(self):
        # Match slowrun's CUDA runtime precision: bf16 embeddings and bf16 norm weights.
        if self.transformer.wte.weight.device.type != "cuda":
            return
        self.transformer.wte.to(dtype=torch.bfloat16)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def init_weights(self):
        _init_trunc_normal_(self.transformer.wte.weight, std=self.config.n_embd ** -1)
        _init_trunc_normal_(self.lm_head.weight, std=_linear_fan_in_std(self.lm_head))
        self.final_norm.weight.fill_(1.0)
        for block in self.transformer.h:
            block.attn_norm.weight.fill_(1.0)
            block.mlp_norm.weight.fill_(1.0)
            block.attn.q_norm.weight.fill_(1.0)
            block.attn.k_norm.weight.fill_(1.0)
            _init_trunc_normal_(block.attn.c_q.weight, std=_linear_fan_in_std(block.attn.c_q))
            _init_trunc_normal_(block.attn.c_k.weight, std=_linear_fan_in_std(block.attn.c_k))
            _init_trunc_normal_(block.attn.c_v.weight, std=_linear_fan_in_std(block.attn.c_v))
            _init_trunc_normal_(block.attn.c_proj.weight, std=_linear_fan_in_std(block.attn.c_proj))
            _init_trunc_normal_(block.mlp.c_gate.weight, std=_linear_fan_in_std(block.mlp.c_gate))
            _init_trunc_normal_(block.mlp.c_fc.weight, std=_linear_fan_in_std(block.mlp.c_fc))
            _init_trunc_normal_(block.mlp.c_proj.weight, std=_linear_fan_in_std(block.mlp.c_proj))
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.apply_runtime_precision_fixup()

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self, window_sizes=None):
        if window_sizes is None:
            window_sizes = self.window_sizes
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.transformer.wte.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) if w[0] >= 0 else 12 * h * q * t for w in window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def forward(self, idx=None, targets=None, loss_reduction='mean', attn_mask=None, loss_mask=None, input_embeds=None, causal=True, window_sizes=None):
        if (idx is None) == (input_embeds is None):
            raise ValueError("exactly one of idx or input_embeds must be provided")
        if idx is not None:
            B, T = idx.size()
            x = self.transformer.wte(idx)
        else:
            B, T, _ = input_embeds.size()
            x = input_embeds.to(self.transformer.wte.weight.dtype)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        if window_sizes is None:
            window_sizes = self.window_sizes if causal or not hasattr(self, "full_window_sizes") else self.full_window_sizes
        for i, block in enumerate(self.transformer.h):
            block_causal = causal if attn_mask is None else False
            x = block(x, cos_sin, window_sizes[i], attn_mask=attn_mask, causal=block_causal)
        x = self.final_norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is not None:
            if loss_mask is None and not self.manual_ignore_index_loss:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction='none',
            ).view(B, T)
            if loss_mask is None:
                if loss_reduction == 'none':
                    return per_tok
                valid = targets != -1
                valid_f = valid.to(per_tok.dtype)
                summed = (per_tok * valid_f).sum()
                if loss_reduction == 'sum':
                    return summed
                if loss_reduction == 'mean':
                    return summed / valid_f.sum().clamp_min(1.0)
                raise ValueError(f"unsupported loss_reduction={loss_reduction}")
            valid = loss_mask & (targets != -1)
            if loss_reduction == 'none':
                return per_tok * valid
            valid_f = valid.to(per_tok.dtype)
            masked_sum = (per_tok * valid_f).sum()
            if loss_reduction == 'sum':
                return masked_sum
            if loss_reduction == 'mean':
                return masked_sum / valid_f.sum().clamp_min(1.0)
            raise ValueError(f"unsupported loss_reduction={loss_reduction}")
        return logits
