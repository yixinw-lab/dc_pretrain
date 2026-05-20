# Data-Constrained Language Model Pretraining

This repository contains the training and data-preparation code for:

> Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws

The experiments study language-model pretraining when compute is high relative to the amount of unique text. The main regularization method in the code is masked-input regularization (MIR): an auxiliary next-token prediction objective on randomly masked inputs, added without changing the autoregressive inference architecture.

## Repository Layout

```text
.
|-- dcp/
|   |-- data.py                         # packed-token dataloader
|   |-- eval.py                         # validation helpers
|   |-- model.py                        # GPT model and attention backend
|   |-- optim.py                        # distributed AdamW optimizer
|   |-- runtime.py                      # distributed, checkpoint, result, parity helpers
|   |-- train4_diffusionauxiliary_impl.py
|   `-- train4_diffusionauxiliary_ddc_impl.py
|-- train4_diffusionauxiliary.py        # compatibility wrapper for the MIR/baseline trainer
|-- train4_diffusionauxiliary_ddc.py    # compatibility wrapper for the DDC trainer
|-- prepare_data_new.py                 # DCLM data preprocessing
|-- scripts/                            # experiment launch scripts/templates
`-- requirements.txt
```

Generated data, checkpoints, logs, result JSONs, and wandb files are intentionally ignored by git.

## Environment

The reference environment used for the reported runs was:

- Python 3.12.12
- PyTorch 2.10.0+cu128
- CUDA 12.8
- 8 NVIDIA H100 GPUs

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The trainer uses `kernels.get_kernel("varunneal/flash-attention-3")` on H100s when available, and falls back to PyTorch scaled-dot-product attention otherwise. For the reference environment, the FlashAttention-3 kernel was loaded through the `kernels` package.

## Preparing Data

The training scripts expect pre-tokenized `.pt` files containing packed GPT-2-tokenized sequences.

Example for a 200M-token DCLM training split with a 10M-token validation split:

```bash
python prepare_data_new.py \
  --train_tokens 200_000_000 \
  --val_tokens 10_000_000 \
  --local_dir dclm_data_200m
```

This writes:

```text
dclm_data_200m/dclm_train.pt
dclm_data_200m/dclm_val.pt
```

When overflow data is used from `mlfoundations/dclm-baseline-1.0`, `prepare_data_new.py` streams `.jsonl.zst` shards through the system `zstd` CLI, so `zstd` must be installed on the machine.

## Training

Use `torchrun` for distributed training. The reference runs used 8 H100 GPUs, so the examples below launch with `--nproc_per_node=8`; adjust that value and the batch-size settings for other hardware. The wrappers at the repository root are kept so old launch commands can still call the same filenames, while the implementation lives under `dcp/`.

Baseline autoregressive training is stage 0 only:

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=8 train4_diffusionauxiliary.py \
  --run text_dclm200m_baseline_257m \
  --input_bin dclm_data_200m/dclm_train.pt \
  --input_val_bin dclm_data_200m/dclm_val.pt \
  --n_layer 12 \
  --n_head 16 \
  --n_kv_head 16 \
  --n_embd 1024 \
  --device-batch-size 16 \
  --num-epochs 16 \
  --adaptive-stage0-epochs 16 \
  --adaptive-stage1-epochs 0 \
  --adam-lr 1e-3 \
  --weight-decay 0.8 \
  --dropout 0.1 \
  --final-eval-size 500000 \
  --save-result results/result_text_dclm200m_baseline_257m.json \
  --save-final-checkpoint checkpoints/text_dclm200m_baseline_257m.pt
```

MIR training uses the same entrypoint, but assigns epochs to stage 1 and configures the masking objective:

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=8 train4_diffusionauxiliary.py \
  --run text_dclm200m_mir_257m \
  --input_bin dclm_data_200m/dclm_train.pt \
  --input_val_bin dclm_data_200m/dclm_val.pt \
  --n_layer 12 \
  --n_head 16 \
  --n_kv_head 16 \
  --n_embd 1024 \
  --device-batch-size 16 \
  --num-epochs 16 \
  --adaptive-stage0-epochs 0 \
  --adaptive-stage1-epochs 16 \
  --stage1-group-size 4 \
  --stage1-num-groups 64 \
  --stage1-full-seq \
  --stage1-mask-sampling uniform_ratio \
  --stage1-mask-ratio-min 0.0 \
  --stage1-mask-ratio-max 0.5 \
  --ntp-loss-downscale 1.0 \
  --adaptive-loss-downscale 0.4 \
  --adam-lr 1e-3 \
  --weight-decay 0.8 \
  --dropout 0.1 \
  --final-eval-size 500000 \
  --save-result results/result_text_dclm200m_mir_257m.json \
  --save-final-checkpoint checkpoints/text_dclm200m_mir_257m.pt
```

The DDC trainer is available through:

```bash
torchrun --standalone --nproc_per_node=8 train4_diffusionauxiliary_ddc.py ...
```

Result JSONs are written with metrics such as `best_val_loss`, `val_loss`, `final_l2r_loss`, and `final_l2r_bpb`.

## Launch Scripts

The `scripts/` directory contains experiment launchers used during development. They are useful as templates for the paper-scale runs, but they contain machine-specific absolute paths and may need editing before use on a new machine. In particular, check the initial `cd ...` line and the data paths before launching.

Example:

```bash
bash scripts/launch_all5_baseline_scalinglaw_200m.sh
```

## Acknowledgements

Parts of this codebase build on prior open-source work:

- `train4_diffusionauxiliary.py` and `prepare_data_new.py` are based on [`qlabs-eng/slowrun`](https://github.com/qlabs-eng/slowrun).
- `train4_diffusionauxiliary_ddc.py` and its implementation under `dcp/` adapt code from [`wmn-231314/diffusion-data-constraint`](https://github.com/wmn-231314/diffusion-data-constraint).
- The DCLM data preparation uses data-generation code from the Marin project, specifically [`marin-community/marin` on the `suhas/data-efficiency` branch](https://github.com/marin-community/marin/tree/suhas/data-efficiency).

## Notes

- `WANDB_MODE=offline` is supported and was used for many local runs.
- `--save-result` writes a compact JSON summary for downstream result tables.
- `--save-final-checkpoint` writes a final checkpoint containing model state, model config, run metadata, and training summary.
