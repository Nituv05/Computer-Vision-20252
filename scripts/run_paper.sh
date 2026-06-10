#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
METHODS="${METHODS:-all}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-pacs vlcs office_home}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-}"
SAVE_DIR="${SAVE_DIR:-outputs/paper30_${BACKBONE/resnet/r}}"
WANDB="${WANDB:-0}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-cv20252}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"

read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

project_name() {
  local dataset="$1"
  echo "${WANDB_PROJECT_PREFIX}-${dataset//_/-}"
}

for dataset in "${DATASET_ARGS[@]}"; do
  dataset_tag="${dataset//_/-}"
  command=(
    python tools/run_experiments.py
    --dataset "${dataset}"
    --data_root "${DATA_ROOT}"
    --methods "${METHOD_ARGS[@]}"
    --backbones "${BACKBONE}"
    --seeds "${SEED_ARGS[@]}"
    --epochs "${EPOCHS}"
    --holdout_fraction 0.2
    --scheduler none
    --hparams_profile paper
    --num_workers "${NUM_WORKERS}"
    --save_dir "${SAVE_DIR}"
    --skip_existing
  )

  if [[ -n "${BATCH_SIZE}" ]]; then
    command+=(--batch_size "${BATCH_SIZE}")
  fi

  case "${WANDB}" in
    1|true|TRUE|yes|YES)
      command+=(
        --wandb
        --wandb_project "$(project_name "${dataset}")"
        --wandb_group "${dataset_tag}-${BACKBONE}-paper30"
        --wandb_tags "${dataset}" "${BACKBONE}" paper30 full
      )
      if [[ -n "${WANDB_ENTITY}" ]]; then
        command+=(--wandb_entity "${WANDB_ENTITY}")
      fi
      if [[ -n "${WANDB_MODE}" ]]; then
        command+=(--wandb_mode "${WANDB_MODE}")
      fi
      ;;
  esac

  echo "Running ${dataset} with ${BACKBONE}, methods=${METHODS}, seeds=${SEEDS}"
  "${command[@]}"
done

if [[ "${SKIP_SUMMARY:-0}" != "1" ]]; then
  python tools/summarize_results.py \
    --metrics_dir "${SAVE_DIR}" \
    --output_csv "outputs/paper30_${BACKBONE/resnet/r}_results.csv"
fi

echo "Finished ${BACKBONE} paper runs."
echo "Checkpoints and metrics: ${SAVE_DIR}"
echo "Summary CSV: outputs/paper30_${BACKBONE/resnet/r}_results.csv"
