#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/data_constrained_pretraining
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate train

mkdir -p logs
mkdir -p checkpoints
mkdir -p results

default_nproc_per_node="${SLURM_GPUS_PER_NODE:-8}"
nproc_per_node="${NPROC_PER_NODE:-$default_nproc_per_node}"
num_epochs="${NUM_EPOCHS:-500}"
eval_every_epochs="${EVAL_EVERY_EPOCHS:-10}"
eval_every_final_epochs="${EVAL_EVERY_FINAL_EPOCHS:-200}"
adam_lr="${ADAM_LR:-2e-4}"
weight_decay="${WEIGHT_DECAY:-3.2}"
final_eval_size="${FINAL_EVAL_SIZE:-10000000}"
wandb_project="${WANDB_PROJECT:-overtrain-dclm}"
wandb_group="${WANDB_GROUP:-ddc-adam-1p4b}"
wandb_mode="${WANDB_MODE:-offline}"
warmup_ratio="${WARMUP_RATIO:-0.01}"
max_grad_norm="${MAX_GRAD_NORM:-1.0}"
mdm_val_num_mc="${MDM_VAL_NUM_MC:-32}"
mdm_final_eval_num_mc="${MDM_FINAL_EVAL_NUM_MC:-32}"
device_batch_size="${DEVICE_BATCH_SIZE:-4}"
total_batch_size="${TOTAL_BATCH_SIZE:-524288}"
input_bin="${INPUT_BIN:-dclm_data/dclm_train.pt}"
input_val_bin="${INPUT_VAL_BIN:-dclm_data/dclm_val.pt}"

format_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

wd_tag="$(format_tag "$weight_decay")"
lr_tag="$(format_tag "$adam_lr")"
timestamp="$(date +%F_%H-%M-%S)"

run_base_name="text_dclm_ddc_adam_1p4b_e${num_epochs}_wd${wd_tag}_lr${lr_tag}_finaleval_${final_eval_size}"
run_name="${run_base_name}_${timestamp}"
result_path="results/result_${run_name}.json"
log_path="logs/${run_name}.log"
checkpoint_path="checkpoints/${run_name}.pt"

echo "[1/1] Starting $run_name"
echo "  epochs=$num_epochs eval_every_epochs=$eval_every_epochs eval_every_final_epochs=$eval_every_final_epochs weight_decay=$weight_decay adam_lr=$adam_lr"
echo "  warmup_ratio=$warmup_ratio max_grad_norm=$max_grad_norm"
echo "  device_batch_size=$device_batch_size total_batch_size=$total_batch_size nproc_per_node=$nproc_per_node"
echo "  mdm_val_num_mc=$mdm_val_num_mc mdm_final_eval_num_mc=$mdm_final_eval_num_mc"
echo "  checkpoint_path=$checkpoint_path"

set +e
WANDB_MODE="$wandb_mode" PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node="$nproc_per_node" train4_diffusionauxiliary_ddc.py \
  --run "$run_name" \
  --training-mode mdm \
  --n_layer 24 \
  --n_head 32 \
  --n_kv_head 32 \
  --n_embd 2048 \
  --device-batch-size "$device_batch_size" \
  --total-batch-size "$total_batch_size" \
  --dropout 0.1 \
  --log-grad-norms \
  --input_bin "$input_bin" \
  --input_val_bin "$input_val_bin" \
  --num-epochs "$num_epochs" \
  --eval-every-epochs "$eval_every_epochs" \
  --eval-every-final-epochs "$eval_every_final_epochs" \
  --adam-lr "$adam_lr" \
  --warmup-ratio "$warmup_ratio" \
  --max-grad-norm "$max_grad_norm" \
  --weight-decay "$weight_decay" \
  --mdm-val-num-mc "$mdm_val_num_mc" \
  --mdm-final-eval-num-mc "$mdm_final_eval_num_mc" \
  --final-eval-size "$final_eval_size" \
  --final-lr-frac 0.1 \
  --wandb_project "$wandb_project" \
  --wandb_group "$wandb_group" \
  --save-final-checkpoint "$checkpoint_path" \
  --save-result "$result_path" \
  2>&1 | tee "$log_path"
run_status=${PIPESTATUS[0]}
set -e

if [[ "$run_status" -ne 0 ]]; then
  echo "[1/1] Run failed with exit code $run_status"
  exit "$run_status"
fi
