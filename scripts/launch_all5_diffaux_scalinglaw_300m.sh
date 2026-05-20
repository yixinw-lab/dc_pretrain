#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/data_constrained_pretraining
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate train

mkdir -p logs
mkdir -p checkpoints
mkdir -p results

nproc_per_node="${NPROC_PER_NODE:-8}"
final_eval_size="${FINAL_EVAL_SIZE:-500000}"
wandb_project="${WANDB_PROJECT:-overtrain-dclm-300m}"
wandb_group_prefix="${WANDB_GROUP_PREFIX:-diffaux-adam-scalinglaw}"
wandb_mode="${WANDB_MODE:-offline}"
warmup_ratio="${WARMUP_RATIO:-0.01}"
max_grad_norm="${MAX_GRAD_NORM:-1.0}"

configs=(
  "1p4b|24|32|32|2048|8|32|1.6|1e-3"
  "664m|18|24|24|1536|16|32|0.8|1e-3"
  "257m|12|16|16|1024|16|32|0.8|1e-3"
  "140m|9|12|12|768|16|64|0.2|3e-3"
  "72m|6|8|8|512|16|64|0.1|1e-2"
)

total_jobs="${#configs[@]}"
current_job=0

format_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

echo "Launching ${total_jobs} diffaux scaling-law runs sequentially."
echo "nproc_per_node=$nproc_per_node final_eval_size=$final_eval_size wandb_mode=$wandb_mode"

for config in "${configs[@]}"; do
  IFS='|' read -r model_tag n_layer n_head n_kv_head n_embd device_batch_size num_epochs weight_decay lr <<< "$config"
  current_job=$((current_job + 1))

  wd_tag="$(format_tag "$weight_decay")"
  lr_tag="$(format_tag "$lr")"
  timestamp="$(date +%F_%H-%M-%S)"

  run_base_name="dclm300m_diffaux_${model_tag}_scalinglaw_e${num_epochs}_wd${wd_tag}_lr${lr_tag}_finaleval_${final_eval_size}"
  run_name="${run_base_name}_${timestamp}"
  result_path="results/result_${run_name}.json"
  log_path="logs/${run_name}.log"
  checkpoint_path="checkpoints/${run_name}.pt"
  wandb_group="${wandb_group_prefix}-${model_tag}"

  echo
  echo "[$current_job/$total_jobs] Starting $run_name"
  echo "  model=$model_tag n_layer=$n_layer n_head=$n_head n_kv_head=$n_kv_head n_embd=$n_embd"
  echo "  device_batch_size=$device_batch_size epochs=$num_epochs weight_decay=$weight_decay adam_lr=$lr"
  echo "  warmup_ratio=$warmup_ratio max_grad_norm=$max_grad_norm"

  set +e
  WANDB_MODE="$wandb_mode" PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$nproc_per_node" train4_diffusionauxiliary.py \
      --run "$run_name" \
      --n_layer "$n_layer" \
      --n_head "$n_head" \
      --n_kv_head "$n_kv_head" \
      --n_embd "$n_embd" \
      --device-batch-size "$device_batch_size" \
      --log-grad-norms \
      --input_bin "dclm_data_300m/dclm_train.pt" \
      --input_val_bin "dclm_data_300m/dclm_val.pt" \
      --num-epochs "$num_epochs" \
      --adaptive-stage0-epochs 0 \
      --adaptive-stage1-epochs "$num_epochs" \
      --adam-lr "$lr" \
      --warmup-ratio "$warmup_ratio" \
      --max-grad-norm "$max_grad_norm" \
      --dropout 0.1 \
      --weight-decay "$weight_decay" \
      --final-eval-size "$final_eval_size" \
      --stage1-right-window-size 0 \
      --stage1-full-seq \
      --ntp-loss-downscale 1.0 \
      --adaptive-loss-downscale 0.4 \
      --stage1-mask-sampling uniform_ratio \
      --stage1-mask-ratio-min 0.0 \
      --stage1-mask-ratio-max 0.5 \
      --wandb_project "$wandb_project" \
      --wandb_group "$wandb_group" \
      --save-final-checkpoint "$checkpoint_path" \
      --save-result "$result_path" \
      2>&1 | tee "$log_path"
  run_status=${PIPESTATUS[0]}
  set -e

  if [[ "$run_status" -ne 0 ]]; then
    echo "[$current_job/$total_jobs] Run failed with exit code $run_status"
    exit "$run_status"
  fi
done

echo
echo "All diffaux scaling-law runs finished."
