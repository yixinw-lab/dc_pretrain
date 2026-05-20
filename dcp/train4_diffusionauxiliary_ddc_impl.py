"""
GPT trainer based on https://github.com/qlabs-eng/slowrun 
and https://github.com/wmn-231314/diffusion-data-constraint

This variant supports both standard causal next-token prediction (`ntp`) and
DDC-style masked diffusion pretraining (`mdm`) while keeping the same fast
training engine.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
import tiktoken
from datetime import timedelta
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    import wandb
except ImportError:
    wandb = None

from dcp.data import DataLoader, _assert_loader_vocab_compatible
from dcp.eval import evaluate_bpb_token_budget, evaluate_mdm_loss_token_budget
from dcp.model import GPT as BaseGPT, GPTConfig, RMSNorm, fa3_available
from dcp.optim import DistShardedAdamW
from dcp.runtime import (
    DummyWandb,
    _hash_model_grads,
    _hash_named_tensors,
    _hash_optimizer_state,
    _hash_rng_states,
    append_parity_record,
    get_dist_info,
    initialize_parity_dump,
    load_final_checkpoint,
    persist_result_json,
    print0,
    save_final_checkpoint,
)

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

def _parse_positive_token_count(value: str) -> int:
    cleaned = value.replace(",", "").replace("_", "")
    try:
        parsed = int(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("final-eval-size must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("final-eval-size must be a positive integer")
    return parsed


parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--num-epochs", type=int, default=16)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--adam-lr", type=float, default=2e-4)
parser.add_argument("--warmup-ratio", type=float, default=0.01)
parser.add_argument("--warmdown-ratio", type=float, default=0.1)
parser.add_argument("--final-lr-frac", type=float, default=0.1)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=524288)
# total_bsize = device_bsize * 8 * 2048 * grad_accum_steps
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=12)
parser.add_argument("--n_head", type=int, default=12)
parser.add_argument("--n_kv_head", type=int, default=12)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_project", type=str, default="overtrain-dclm")
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--log-grad-norms", action="store_true")
parser.add_argument("--mask-token-id", type=int, default=None)
parser.add_argument("--final-eval-size", type=_parse_positive_token_count, default=None)
parser.add_argument("--save-final-checkpoint", type=str, default="")
parser.add_argument("--load-final-checkpoint", type=str, default="")
parser.add_argument("--eval-only-final", action="store_true")
parser.add_argument("--ntp-loss-downscale", type=float, default=1.0)
parser.add_argument("--training-mode", type=str, default="mdm", choices=["mdm", "ntp"])
parser.add_argument("--mdm-mask-eps", type=float, default=1e-3)
parser.add_argument("--mdm-ar-mask-ratio", type=float, default=0.0)
parser.add_argument("--mdm-val-num-mc", type=int, default=1)
parser.add_argument("--mdm-final-eval-num-mc", type=int, default=None)
parser.add_argument("--eval-every-epochs", type=int, default=1)
parser.add_argument("--eval-every-final-epochs", type=int, default=0)
parser.add_argument("--max-train-steps", type=int, default=0)
parser.add_argument("--parity-dump", type=str, default="")
args = parser.parse_args()

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json
if args.eval_only_final and not args.load_final_checkpoint:
    raise ValueError("--eval-only-final requires --load-final-checkpoint")
if args.n_layer <= 0:
    raise ValueError("--n_layer must be positive")
if args.n_head <= 0:
    raise ValueError("--n_head must be positive")
if args.n_kv_head <= 0:
    raise ValueError("--n_kv_head must be positive")
if args.n_embd <= 0:
    raise ValueError("--n_embd must be positive")
if args.n_embd % args.n_head != 0:
    raise ValueError("--n_embd must be divisible by --n_head")
if args.n_head % args.n_kv_head != 0:
    raise ValueError("--n_head must be divisible by --n_kv_head")
if args.adam_lr <= 0:
    raise ValueError("--adam-lr must be positive")
if not (0.0 <= args.warmup_ratio <= 1.0):
    raise ValueError("--warmup-ratio must be in [0, 1]")
if not (0.0 <= args.warmdown_ratio <= 1.0):
    raise ValueError("--warmdown-ratio must be in [0, 1]")
if not (0.0 <= args.final_lr_frac <= 1.0):
    raise ValueError("--final-lr-frac must be in [0, 1]")
if args.max_grad_norm <= 0:
    raise ValueError("--max-grad-norm must be positive")
if not (0.0 < args.mdm_mask_eps <= 1.0):
    raise ValueError("--mdm-mask-eps must be in (0, 1]")
if not (0.0 <= args.mdm_ar_mask_ratio < 1.0):
    raise ValueError("--mdm-ar-mask-ratio must be in [0, 1)")
if args.mdm_val_num_mc <= 0:
    raise ValueError("--mdm-val-num-mc must be positive")
if args.mdm_final_eval_num_mc is not None and args.mdm_final_eval_num_mc <= 0:
    raise ValueError("--mdm-final-eval-num-mc must be positive when provided")
if args.eval_every_epochs <= 0:
    raise ValueError("--eval-every-epochs must be positive")
if args.eval_every_final_epochs < 0:
    raise ValueError("--eval-every-final-epochs must be non-negative")
if args.max_train_steps < 0:
    raise ValueError("--max-train-steps must be non-negative")


def _compute_mlp_hidden_dim(n_embd: int) -> int:
    return 256 * ((8 * n_embd // 3 + 255) // 256)


def should_evaluate_epoch(epoch: int, num_epochs: int, eval_every_epochs: int, eval_every_final_epochs: int) -> bool:
    if epoch == num_epochs:
        return True
    if eval_every_final_epochs > 0 and epoch > num_epochs - eval_every_final_epochs:
        return True
    return epoch % eval_every_epochs == 0

# =============================================================================
# Hyperparameters
# =============================================================================

# Architecture
DEPTH = args.n_layer if args.n_layer is not None else 12
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 12
N_KV_HEAD = args.n_kv_head if args.n_kv_head is not None else 12
HEAD_DIM = N_EMBD // N_HEAD
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = 10_000_000
DATA_DIR = "dclm_data"

# Base optimizer hyperparameters
BASE_ADAM_LR = args.adam_lr

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
ADAM_LR = BASE_ADAM_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WARMUP_RATIO = args.warmup_ratio
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = args.final_lr_frac
MAX_GRAD_NORM = args.max_grad_norm

NTP_LOSS_DOWNSCALE = args.ntp_loss_downscale
assert NTP_LOSS_DOWNSCALE >= 0.0 and NTP_LOSS_DOWNSCALE <= 1.0, f"Value Error: NTP_LOSS_DOWNSCALE must be in [0.0, 1.0]"
TRAINING_MODE = args.training_mode
MDM_MASK_EPS = args.mdm_mask_eps
MDM_AR_MASK_RATIO = args.mdm_ar_mask_ratio
MDM_VAL_NUM_MC = args.mdm_val_num_mc
MDM_FINAL_EVAL_NUM_MC = args.mdm_final_eval_num_mc if args.mdm_final_eval_num_mc is not None else args.mdm_val_num_mc
# =============================================================================
# Utilities
# =============================================================================

# Runtime helpers are imported from dcp.runtime.

# =============================================================================
# GPT Model
# =============================================================================

class GPT(BaseGPT):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(
            config,
            pad_vocab_size_to=pad_vocab_size_to,
            use_full_window_sizes=True, manual_ignore_index_loss=True,
        )

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
            window_sizes = self.window_sizes if causal else self.full_window_sizes
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, window_sizes[i], attn_mask=attn_mask, causal=causal)
        x = self.final_norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if targets is not None:
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

    def setup_optimizer(self):
        no_decay = set()
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, RMSNorm)):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    no_decay.add(full_name)
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or name in no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = DistShardedAdamW(
            [
                {"params": decay_params, "lr": ADAM_LR, "betas": ADAM_BETAS, "eps": ADAM_EPS, "weight_decay": WEIGHT_DECAY},
                {"params": no_decay_params, "lr": ADAM_LR, "betas": ADAM_BETAS, "eps": ADAM_EPS, "weight_decay": 0.0},
            ],
            max_grad_norm=MAX_GRAD_NORM,
            return_grad_norm=args.log_grad_norms or args.parity_dump,
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

# Optimizer helpers are imported from dcp.optim.

# Data loading helpers are imported from dcp.data.

# =============================================================================
# Loss evaluation
# =============================================================================

# Eval helpers are imported from dcp.eval.

def build_random_mask_ddc_batch(clean_tokens, mask_token_id, mask_eps, ar_mask_ratio=0.0):
    batch_size, seq_len = clean_tokens.shape
    device = clean_tokens.device
    valid_positions = clean_tokens != -1

    sampled_t = torch.rand(batch_size, device=device)
    p_mask_scalar = (1.0 - mask_eps) * sampled_t + mask_eps
    p_mask = p_mask_scalar[:, None].expand(batch_size, seq_len)

    masked_positions = (torch.rand((batch_size, seq_len), device=device) < p_mask) & valid_positions

    if ar_mask_ratio > 0.0:
        ar_rows = torch.rand(batch_size, device=device) <= ar_mask_ratio
        for row_idx in range(batch_size):
            if not ar_rows[row_idx]:
                continue
            valid_count = int(valid_positions[row_idx].sum().item())
            if valid_count <= 0:
                continue
            num_masked = int(masked_positions[row_idx].sum().item())
            num_masked = min(num_masked, valid_count)
            masked_positions[row_idx] = False
            if num_masked > 0:
                start = valid_count - num_masked
                masked_positions[row_idx, start:valid_count] = True

    noisy_tokens = clean_tokens.clone()
    noisy_tokens[masked_positions] = mask_token_id

    total_valid = valid_positions.sum().clamp_min(1)
    actual_mask_ratio = masked_positions.sum().to(torch.float32) / total_valid.to(torch.float32)
    return {
        "noisy_tokens": noisy_tokens,
        "targets": clean_tokens,
        "masked_positions": masked_positions,
        "p_mask": p_mask,
        "sampled_t": sampled_t,
        "actual_mask_ratio": actual_mask_ratio,
        "p_mask_mean": p_mask_scalar.mean(),
    }


def compute_ddc_loss(model, clean_tokens, mask_token_id, mask_eps, ar_mask_ratio=0.0):
    ddc_batch = build_random_mask_ddc_batch(
        clean_tokens,
        mask_token_id=mask_token_id,
        mask_eps=mask_eps,
        ar_mask_ratio=ar_mask_ratio,
    )
    per_tok = model(
        ddc_batch["noisy_tokens"],
        ddc_batch["targets"],
        loss_reduction='none',
        causal=False,
    )
    masked = ddc_batch["masked_positions"]
    weighted = (per_tok * masked.to(per_tok.dtype)) / ddc_batch["p_mask"]
    loss = weighted.sum() / (clean_tokens.size(0) * clean_tokens.size(1))
    return loss, ddc_batch


# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=7))
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# GPU info for MFU
gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name: gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

# FA3 status
if fa3_available():
    print0("Using Flash Attention 3 (Hopper GPU detected) for causal batches")
else:
    print0("Using PyTorch SDPA fallback (no FA3)")

# Optional checkpoint load for eval-only final runs
checkpoint = load_final_checkpoint(args.load_final_checkpoint) if args.eval_only_final else None
checkpoint_args = checkpoint.get("args", {}) if checkpoint else {}
checkpoint_summary = checkpoint.get("training_summary", {}) if checkpoint else {}
metadata_args = checkpoint_args if checkpoint else vars(args)
loaded_run_name = checkpoint.get("run") if checkpoint else None
checkpoint_objective_version = int(checkpoint_summary.get("ddc_objective_version", 0)) if checkpoint else 0

if checkpoint:
    if checkpoint_objective_version >= 1:
        active_training_mode = checkpoint_args.get("training_mode", TRAINING_MODE)
    else:
        active_training_mode = "ntp"
        if checkpoint_args.get("training_mode") == "mdm":
            print0("Checkpoint predates DDC MDM implementation; treating it as ntp for eval-only-final.")
else:
    active_training_mode = TRAINING_MODE

if active_training_mode not in {"mdm", "ntp"}:
    raise ValueError(f"unsupported training mode: {active_training_mode}")

active_mdm_mask_eps = float(checkpoint_args.get("mdm_mask_eps", MDM_MASK_EPS)) if args.eval_only_final else MDM_MASK_EPS
active_mdm_ar_mask_ratio = float(checkpoint_args.get("mdm_ar_mask_ratio", MDM_AR_MASK_RATIO)) if args.eval_only_final else MDM_AR_MASK_RATIO
active_mdm_val_num_mc = int(checkpoint_args.get("mdm_val_num_mc", MDM_VAL_NUM_MC)) if args.eval_only_final else MDM_VAL_NUM_MC
active_mdm_final_eval_num_mc = int(checkpoint_args.get("mdm_final_eval_num_mc", MDM_FINAL_EVAL_NUM_MC)) if args.eval_only_final else MDM_FINAL_EVAL_NUM_MC
if active_mdm_val_num_mc <= 0:
    raise ValueError("active mdm validation MC count must be positive")
if active_mdm_final_eval_num_mc <= 0:
    raise ValueError("active mdm final eval MC count must be positive")
if args.eval_only_final:
    checkpoint_device_batch_size = int(checkpoint_args.get("device_batch_size", args.device_batch_size))
    active_device_batch_size = max(1, min(args.device_batch_size, checkpoint_device_batch_size))
else:
    active_device_batch_size = args.device_batch_size

# wandb
if args.run:
    run_name = args.run
elif args.eval_only_final and loaded_run_name:
    run_name = f"{loaded_run_name}_evalonly"
else:
    run_name = time.strftime("%Y%m%d_%H%M%S")
_wandb_kwargs = {"project": args.wandb_project, "name": run_name}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if (not master_process or wandb is None) else wandb.init(**_wandb_kwargs)
if master_process and wandb is not None:
    wandb_run.log_code(".")

# Load the original slowrun GPT-2 tokenizer and compute token_bytes for BPB evaluation.
encoder = tiktoken.get_encoding("gpt2")
base_vocab_size = encoder.n_vocab
checkpoint_mask_token_id = checkpoint.get("mask_token_id") if checkpoint else None
if checkpoint:
    model_config = dict(checkpoint["model_config"])
    model_config.pop("mlp_dim", None)
    config = GPTConfig(**model_config)
    if checkpoint_mask_token_id is not None:
        mask_token_id = int(checkpoint_mask_token_id)
    elif config.vocab_size > base_vocab_size:
        mask_token_id = config.vocab_size - 1
    else:
        mask_token_id = None
else:
    mask_token_id = args.mask_token_id if args.mask_token_id is not None else base_vocab_size
    vocab_size = max(base_vocab_size, mask_token_id + 1)
    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        n_embd=N_EMBD,
        window_pattern=WINDOW_PATTERN,
        dropout=args.dropout,
    )
vocab_size = config.vocab_size
if active_training_mode == "mdm" and mask_token_id is None:
    raise ValueError("mdm mode requires a mask token id")
if active_training_mode == "mdm" and NTP_LOSS_DOWNSCALE > 0.0:
    print0("Ignoring --ntp-loss-downscale in mdm mode; mdm training is diffusion-only.")

# Print hyperparameters
print0(f"--- Hyperparameters ---")
print0(f"  mode={'eval_only_final' if args.eval_only_final else 'train_and_eval'}")
print0(f"  training_mode={active_training_mode}")
print0(
    f"  n_layer={config.n_layer}, n_embd={config.n_embd}, mlp_dim={_compute_mlp_hidden_dim(config.n_embd)}, "
    f"n_head={config.n_head}, head_dim={config.n_embd // config.n_head}"
)
print0(f"  seq_len={config.sequence_len}, window_pattern={config.window_pattern}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={active_device_batch_size}")
print0(f"  adam_lr={ADAM_LR}, lr_multiplier={args.lr_multiplier}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}, adam_eps={ADAM_EPS}")
print0(
    f"  warmup_ratio={WARMUP_RATIO}, scheduler=linear_warmup_cosine_decay, warmdown_ratio(unused)={WARMDOWN_RATIO}, "
    f"final_lr_frac={FINAL_LR_FRAC}, max_grad_norm={MAX_GRAD_NORM}"
)
print0(f"  wandb_project={args.wandb_project}, wandb_group={args.wandb_group}")
print0(f"  num_epochs={metadata_args.get('num_epochs', args.num_epochs)}, patience={args.patience}")
print0(f"  final_eval_size_requested={args.final_eval_size if args.final_eval_size is not None else EVAL_TOKENS}")
print0(f"  dropout={config.dropout}")
print0(
    f"  mdm_mask_eps={active_mdm_mask_eps}, "
    f"mdm_ar_mask_ratio={active_mdm_ar_mask_ratio}, "
    f"mdm_val_num_mc={active_mdm_val_num_mc}, "
    f"mdm_final_eval_num_mc={active_mdm_final_eval_num_mc}, "
    f"eval_every_epochs={args.eval_every_epochs}, "
    f"eval_every_final_epochs={args.eval_every_final_epochs}"
)
print0(
    f"  ntp_loss_downscale={metadata_args.get('ntp_loss_downscale', NTP_LOSS_DOWNSCALE)}"
)
print0(f"-----------------------")
print0(
    f"Base vocab size: {base_vocab_size:,} | model vocab size: {vocab_size:,} | "
    f"mask_token_id={mask_token_id}"
)
if checkpoint:
    print0(f"Loaded final checkpoint metadata from {args.load_final_checkpoint}")

eot_id = encoder._special_tokens["<|endoftext|>"]
token_bytes_list = []
for token_id in range(vocab_size):
    if token_id == eot_id or token_id >= encoder.n_vocab:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(encoder.decode_single_token_bytes(token_id)))
token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

# Build model
with torch.device("meta"):
    orig_model = GPT(config)
orig_model.to_empty(device=device)
orig_model.init_weights()
if checkpoint:
    load_info = orig_model.load_state_dict(checkpoint["model_state"], strict=True)
    if load_info.missing_keys or load_info.unexpected_keys:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
        )
orig_model.apply_runtime_precision_fixup()

param_counts = sum(p.numel() for p in orig_model.parameters())
transformer_params = sum(p.numel() for p in orig_model.transformer.h.parameters())
lm_head_params = sum(p.numel() for p in orig_model.lm_head.parameters())
other_params = param_counts - transformer_params - lm_head_params
flop_window_sizes = orig_model.full_window_sizes if active_training_mode == "mdm" else None
num_flops_per_token = orig_model.estimate_flops(window_sizes=flop_window_sizes)
print0(f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})")
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Compile the full model, keeping an unwrapped reference for checkpoints.
model = torch.compile(orig_model, dynamic=False)

# Shared dataloaders / evaluation config
_train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "dclm_train.pt")
_val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "dclm_val.pt")
def build_val_loader():
    loader = DataLoader(_val_path, active_device_batch_size, MAX_SEQ_LEN, device=device)
    _assert_loader_vocab_compatible(loader, config.vocab_size - 1, "validation")
    return loader


def _resolve_eval_token_budget(loader, requested_tokens):
    target_tokens = EVAL_TOKENS if requested_tokens is None else requested_tokens
    if loader.total_tokens <= 0:
        raise ValueError(
            "validation/eval loader has zero usable tokens; reduce --device-batch-size "
            "or regenerate the packed dataset with more rows for this split."
        )
    return min(target_tokens, loader.total_tokens)


_val_loader_probe = build_val_loader()
default_eval_token_budget = _resolve_eval_token_budget(_val_loader_probe, None)
final_eval_token_budget = _resolve_eval_token_budget(_val_loader_probe, args.final_eval_size)
del _val_loader_probe

# Training/eval bookkeeping
step = int(checkpoint.get("step", 0)) if checkpoint else 0
current_epoch = int(checkpoint.get("current_epoch", 0)) if checkpoint else 0
val_loss = checkpoint_summary.get("last_val_loss")
min_val_bpb = checkpoint_summary.get("best_val_bpb", float("inf"))
min_val_loss = checkpoint_summary.get("best_val_loss", float("inf"))
val_mdm_loss = checkpoint_summary.get("last_val_mdm_loss")
best_val_mdm_loss = checkpoint_summary.get("best_val_mdm_loss", min_val_loss if active_training_mode == "mdm" else float("inf"))
epochs_without_improvement = 0
smooth_train_loss = 0.0
total_training_time = float(checkpoint_summary.get("total_training_time", 0.0))
final_train_loss = checkpoint_summary.get("final_train_loss", float("nan"))

result_payload = {
    "run": run_name,
    "adam_lr": metadata_args.get("adam_lr", args.adam_lr),
    "lr_multiplier": metadata_args.get("lr_multiplier", args.lr_multiplier),
    "warmup_ratio": metadata_args.get("warmup_ratio", args.warmup_ratio),
    "warmdown_ratio": metadata_args.get("warmdown_ratio", args.warmdown_ratio),
    "final_lr_frac": metadata_args.get("final_lr_frac", args.final_lr_frac),
    "max_grad_norm": metadata_args.get("max_grad_norm", args.max_grad_norm),
    "weight_decay": metadata_args.get("weight_decay", args.weight_decay),
    "num_epochs": metadata_args.get("num_epochs", args.num_epochs),
    "final_eval_size": final_eval_token_budget,
    "ntp_loss_downscale": metadata_args.get("ntp_loss_downscale", NTP_LOSS_DOWNSCALE),
    "training_mode": active_training_mode,
    "device_batch_size": active_device_batch_size,
    "mdm_mask_eps": active_mdm_mask_eps,
    "mdm_ar_mask_ratio": active_mdm_ar_mask_ratio,
    "mdm_val_num_mc": active_mdm_val_num_mc,
    "mdm_final_eval_num_mc": active_mdm_final_eval_num_mc,
    "eval_every_epochs": args.eval_every_epochs,
    "eval_every_final_epochs": args.eval_every_final_epochs,
    "wandb_url": getattr(wandb_run, "url", None),
    "wandb_project": args.wandb_project,
    "wandb_group": args.wandb_group,
}
if args.eval_only_final:
    result_payload["load_final_checkpoint"] = args.load_final_checkpoint
if val_loss is not None:
    result_payload["val_loss"] = val_loss
if math.isfinite(min_val_loss):
    result_payload["best_val_loss"] = min_val_loss
if val_mdm_loss is not None:
    result_payload["val_mdm_loss"] = val_mdm_loss
if math.isfinite(best_val_mdm_loss):
    result_payload["best_val_mdm_loss"] = best_val_mdm_loss

optimizer = None
train_loader = None
grad_accum_steps = None
num_iterations = None
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
if not args.eval_only_final:
    optimizer = model.setup_optimizer()
    train_loader = DataLoader(_train_path, args.device_batch_size, MAX_SEQ_LEN, device=device)
    _assert_loader_vocab_compatible(train_loader, config.vocab_size - 1, "training")
    if train_loader.total_tokens <= 0:
        raise ValueError(
            "training loader has zero usable tokens; reduce --device-batch-size "
            "or regenerate the packed training dataset with more rows."
        )
    TOKENS_PER_EPOCH = train_loader.total_tokens
    x, y, current_epoch = next(train_loader)
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)  # estimate for LR schedule
    print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
    print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
else:
    print0("Eval-only final mode: skipping training loop and loading weights from checkpoint")
print0(f"Eval set: {default_eval_token_budget:,} tokens (requested {EVAL_TOKENS:,})")
print0(
    f"Final eval budget: {final_eval_token_budget:,} tokens "
    f"(requested {args.final_eval_size if args.final_eval_size is not None else EVAL_TOKENS:,})"
)

# Schedulers
def get_lr_scale(it):
    if num_iterations is None or num_iterations <= 0:
        return 1.0
    warmup = max(1, round(WARMUP_RATIO * num_iterations)) if WARMUP_RATIO > 0 else 0
    warmup = min(warmup, num_iterations)
    if warmup > 0 and it < warmup:
        return (it + 1) / warmup
    if num_iterations <= warmup:
        return 1.0
    decay_span = num_iterations - warmup
    if decay_span <= 1:
        return FINAL_LR_FRAC
    progress = (it - warmup) / (decay_span - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine

wall_clock_start = time.time()
initialize_parity_dump(args.parity_dump)
if not args.eval_only_final:
    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    if active_training_mode == "mdm":
        with autocast_ctx:
            val_mdm_loss = evaluate_mdm_loss_token_budget(
                model,
                val_loader,
                default_eval_token_budget,
                mask_token_id,
                mask_eps=active_mdm_mask_eps,
                compute_ddc_loss_fn=compute_ddc_loss,
                num_mc=active_mdm_val_num_mc,
            )
        val_loss = val_mdm_loss
        best_val_mdm_loss = val_mdm_loss
        min_val_loss = val_mdm_loss
        print0(f"Step {step:05d} | Val MDM Loss: {val_mdm_loss:.6f}")
        wandb_run.log({"step": step, "val/mdm_loss": val_mdm_loss})
        result_payload["val_loss"] = val_mdm_loss
        result_payload["val_mdm_loss"] = val_mdm_loss
        result_payload["best_val_loss"] = min_val_loss
        result_payload["best_val_mdm_loss"] = best_val_mdm_loss
    else:
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb_token_budget(model, val_loader, default_eval_token_budget, token_bytes)
        print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
        min_val_bpb = val_bpb
        min_val_loss = val_loss
        result_payload["val_loss"] = val_loss
        result_payload["best_val_loss"] = min_val_loss
    model.train()

    while current_epoch <= args.num_epochs:
        # Training step
        synchronize()
        t0 = time.time()
        micro_loss_sum = 0.0
        ntp_micro_loss_sum = 0.0
        ntp_micro_count = 0
        mdm_micro_loss_sum = 0.0
        mdm_micro_count = 0
        mdm_mask_ratio_sum = 0.0
        mdm_p_mask_mean_sum = 0.0
        for _ in range(grad_accum_steps):
            if active_training_mode == "mdm":
                with autocast_ctx:
                    mdm_loss, ddc_batch = compute_ddc_loss(
                        model,
                        x,
                        mask_token_id=mask_token_id,
                        mask_eps=MDM_MASK_EPS,
                        ar_mask_ratio=MDM_AR_MASK_RATIO,
                    )
                (mdm_loss / grad_accum_steps).backward()
                raw_mdm_loss = mdm_loss.detach().item()
                micro_loss_sum += raw_mdm_loss
                mdm_micro_loss_sum += raw_mdm_loss
                mdm_micro_count += 1
                mdm_mask_ratio_sum += ddc_batch["actual_mask_ratio"].item()
                mdm_p_mask_mean_sum += ddc_batch["p_mask_mean"].item()
            else:
                raw_ntp_loss = None
                if NTP_LOSS_DOWNSCALE > 0.0:
                    with autocast_ctx:
                        ntp_loss = model(x, y)
                    (ntp_loss * NTP_LOSS_DOWNSCALE / grad_accum_steps).backward()
                    raw_ntp_loss = ntp_loss.detach().item()
                    ntp_micro_loss_sum += raw_ntp_loss
                    ntp_micro_count += 1
                if raw_ntp_loss is not None:
                    micro_loss_sum += raw_ntp_loss * NTP_LOSS_DOWNSCALE
            x, y, epoch = next(train_loader)

        # Update optimizer
        parity_grad_hash = _hash_model_grads(orig_model) if args.parity_dump else None
        lrm = get_lr_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        grad_norm_stats = optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss_f = micro_loss_sum / grad_accum_steps
        raw_ntp_loss_avg = ntp_micro_loss_sum / ntp_micro_count if ntp_micro_count > 0 else None
        raw_mdm_loss_avg = mdm_micro_loss_sum / mdm_micro_count if mdm_micro_count > 0 else None
        mdm_mask_ratio_avg = mdm_mask_ratio_sum / mdm_micro_count if mdm_micro_count > 0 else None
        mdm_p_mask_mean_avg = mdm_p_mask_mean_sum / mdm_micro_count if mdm_micro_count > 0 else None
        synchronize()
        dt = time.time() - t0

        step += 1

        if args.parity_dump:
            append_parity_record(args.parity_dump, {
                "kind": "train_step",
                "rank": ddp_rank,
                "step": step,
                "epoch": int(epoch),
                "current_epoch": int(current_epoch),
                "training_mode": active_training_mode,
                "lr_scale": lrm,
                "train_loss": train_loss_f,
                "raw_ntp_loss": raw_ntp_loss_avg,
                "raw_mdm_loss": raw_mdm_loss_avg,
                "mdm_mask_ratio": mdm_mask_ratio_avg,
                "mdm_p_mask_mean": mdm_p_mask_mean_avg,
                "grad_norm": None if grad_norm_stats is None else grad_norm_stats.get("grad_norm"),
                "grad_hash": parity_grad_hash,
                "model_hash": _hash_named_tensors(orig_model.state_dict().items()),
                "optimizer_hash": _hash_optimizer_state(optimizer),
                "rng": _hash_rng_states(),
            })

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**step)
        pct = 100 * step / num_iterations
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
        if step > 3:
            total_training_time += dt
        steps_done = step - 3
        eta_str = f" | eta: {(num_iterations - step) * total_training_time / steps_done / 60:.1f}m" if steps_done > 0 else ""
        grad_norm_str = ""
        log_payload = {"step": step, "train/mfu": mfu}
        if active_training_mode == "mdm":
            log_payload["train/mdm_loss"] = debiased
            if mdm_mask_ratio_avg is not None:
                log_payload["train/mask_ratio_actual"] = mdm_mask_ratio_avg
            if mdm_p_mask_mean_avg is not None:
                log_payload["train/p_mask_mean"] = mdm_p_mask_mean_avg
        else:
            log_payload["train/loss"] = debiased
            if raw_ntp_loss_avg is not None:
                log_payload["train/raw_loss_normal"] = raw_ntp_loss_avg
        if grad_norm_stats is not None:
            grad_norm_str = f" | grad_norm: {grad_norm_stats['grad_norm']:.4f}"
            log_payload.update({
                "train/grad_norm": grad_norm_stats["grad_norm"],
            })
        metric_name = "mdm_loss" if active_training_mode == "mdm" else "loss"
        print0(f"step {step:05d} ({pct:.2f}%) | {metric_name}: {debiased:.6f}{grad_norm_str} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{eta_str}")
        wandb_run.log(log_payload)

        # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
        if ddp:
            epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
            dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
            epoch = epoch_tensor.item()

        # Epoch boundary: evaluate when the dataloader advances to a new epoch
        if epoch != current_epoch:
            should_run_epoch_eval = should_evaluate_epoch(
                current_epoch,
                args.num_epochs,
                args.eval_every_epochs,
                args.eval_every_final_epochs,
            )
            if should_run_epoch_eval:
                model.eval()
                val_loader = build_val_loader()
                if active_training_mode == "mdm":
                    epoch_val_num_mc = active_mdm_final_eval_num_mc if current_epoch == args.num_epochs else active_mdm_val_num_mc
                    with autocast_ctx:
                        val_mdm_loss = evaluate_mdm_loss_token_budget(
                            model,
                            val_loader,
                            default_eval_token_budget,
                            mask_token_id,
                            mask_eps=active_mdm_mask_eps,
                            compute_ddc_loss_fn=compute_ddc_loss,
                            num_mc=epoch_val_num_mc,
                        )
                    val_loss = val_mdm_loss
                    print0(f"Step {step:05d} | Epoch {current_epoch} | Val MDM Loss: {val_mdm_loss:.6f} | mc: {epoch_val_num_mc}")
                    wandb_run.log({"step": step, "epoch": current_epoch, "val/mdm_loss": val_mdm_loss, "val/mdm_mc": epoch_val_num_mc})
                    result_payload["val_loss"] = val_mdm_loss
                    result_payload["val_mdm_loss"] = val_mdm_loss
                    if val_mdm_loss < min_val_loss:
                        min_val_loss = val_mdm_loss
                        best_val_mdm_loss = val_mdm_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if args.patience >= 0 and epochs_without_improvement >= args.patience:
                            print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                            break
                    result_payload["best_val_loss"] = min_val_loss
                    result_payload["best_val_mdm_loss"] = best_val_mdm_loss
                else:
                    with autocast_ctx:
                        val_bpb, val_loss = evaluate_bpb_token_budget(model, val_loader, default_eval_token_budget, token_bytes)
                    print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
                    wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss})
                    result_payload["val_loss"] = val_loss
                    if val_bpb < min_val_bpb:
                        min_val_bpb = val_bpb
                        min_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if args.patience >= 0 and epochs_without_improvement >= args.patience:
                            print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                            break
                    result_payload["best_val_loss"] = min_val_loss
                model.train()
            else:
                print0(
                    f"Step {step:05d} | Epoch {current_epoch} | "
                    f"Skipping periodic eval "
                    f"(eval_every_epochs={args.eval_every_epochs}, "
                    f"eval_every_final_epochs={args.eval_every_final_epochs})"
                )
            current_epoch = epoch

        # GC management
        if step == 1:
            gc.collect(); gc.freeze(); gc.disable()
        if args.max_train_steps and step >= args.max_train_steps:
            print0(f"Stopping after {step} train step(s) (--max-train-steps={args.max_train_steps})")
            break

    final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
    checkpoint_summary = {
        "ddc_objective_version": 1,
        "last_val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "final_train_loss": final_train_loss,
        "total_training_time": total_training_time,
    }
    if active_training_mode == "mdm":
        checkpoint_summary["last_val_mdm_loss"] = val_mdm_loss
        checkpoint_summary["best_val_mdm_loss"] = best_val_mdm_loss
    else:
        checkpoint_summary["best_val_bpb"] = min_val_bpb
    result_payload["val_loss"] = val_loss
    result_payload["best_val_loss"] = min_val_loss
    if val_mdm_loss is not None:
        result_payload["val_mdm_loss"] = val_mdm_loss
    if math.isfinite(best_val_mdm_loss):
        result_payload["best_val_mdm_loss"] = best_val_mdm_loss
    if args.save_final_checkpoint:
        if master_process:
            save_final_checkpoint(
                args.save_final_checkpoint,
                orig_model,
                mask_token_id,
                run_name,
                step,
                current_epoch,
                vars(args),
                extra_summary=checkpoint_summary,
            )
            result_payload["final_checkpoint_path"] = args.save_final_checkpoint
            persist_result_json(args.save_result, result_payload)
            print0(f"Saved final pre-eval checkpoint to {args.save_final_checkpoint}")
        if ddp:
            dist.barrier()
else:
    persist_result_json(args.save_result, result_payload)

# Final evaluation after training
model.eval()
orig_model.eval()
final_mdm_val_loss = float("nan")
with autocast_ctx:
    if active_training_mode == "mdm":
        final_mdm_loader = build_val_loader()
        final_mdm_val_loss = evaluate_mdm_loss_token_budget(
            model,
            final_mdm_loader,
            final_eval_token_budget,
            mask_token_id,
            mask_eps=active_mdm_mask_eps,
            compute_ddc_loss_fn=compute_ddc_loss,
            num_mc=active_mdm_final_eval_num_mc,
        )
    final_l2r_loader = build_val_loader()
    final_l2r_bpb, final_l2r_loss = evaluate_bpb_token_budget(
        model,
        final_l2r_loader,
        final_eval_token_budget,
        token_bytes,
    )
if active_training_mode == "mdm":
    print0(f"Final Eval | MDM Loss: {final_mdm_val_loss:.6f}")
    wandb_run.log({"final_eval/mdm_loss": final_mdm_val_loss})
    result_payload["final_mdm_val_loss"] = final_mdm_val_loss
print0(f"Final Eval | L2R BPB: {final_l2r_bpb:.6f} | L2R Loss: {final_l2r_loss:.6f}")
wandb_run.log({"final_eval/l2r_bpb": final_l2r_bpb, "final_eval/l2r_loss": final_l2r_loss})
result_payload.update({"final_l2r_bpb": final_l2r_bpb, "final_l2r_loss": final_l2r_loss})
persist_result_json(args.save_result, result_payload)

# Summary
wall_clock_time = time.time() - wall_clock_start
print0(f"Wall clock time: {wall_clock_time/60:.2f}m")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Final train loss: {final_train_loss:.6f}")
if active_training_mode == "mdm":
    if val_mdm_loss is not None:
        print0(f"Last val MDM loss: {val_mdm_loss:.6f}")
    print0(f"Best val MDM loss: {best_val_mdm_loss:.6f}")
    if math.isfinite(final_mdm_val_loss):
        print0(f"Final MDM val loss: {final_mdm_val_loss:.6f}")
else:
    print0(f"Min val BPB: {min_val_bpb:.6f}")
    print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss
wandb_run.summary["final_l2r_loss"] = final_l2r_loss
if active_training_mode == "mdm":
    if val_mdm_loss is not None:
        wandb_run.summary["val_mdm_loss"] = val_mdm_loss
    if math.isfinite(best_val_mdm_loss):
        wandb_run.summary["best_val_mdm_loss"] = best_val_mdm_loss
    if math.isfinite(final_mdm_val_loss):
        wandb_run.summary["final_mdm_val_loss"] = final_mdm_val_loss
result_payload.update({
    "final_train_loss": final_train_loss,
    "best_val_loss": min_val_loss,
    "final_l2r_loss": final_l2r_loss,
})
if val_loss is not None:
    result_payload["val_loss"] = val_loss
if val_mdm_loss is not None:
    result_payload["val_mdm_loss"] = val_mdm_loss
if math.isfinite(best_val_mdm_loss):
    result_payload["best_val_mdm_loss"] = best_val_mdm_loss
if math.isfinite(final_mdm_val_loss):
    result_payload["final_mdm_val_loss"] = final_mdm_val_loss
persist_result_json(args.save_result, result_payload)
if args.save_result and master_process:
    print0(f"Result saved to {args.save_result}")

total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
    
