#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data_root}"
BACKBONE="${BACKBONE:-resnet18}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/checkpoints}"
METHODS="${METHODS:-all}"
DATASETS="${DATASETS:-pacs vlcs office_home}"
EPOCHS="${EPOCHS:-}"
STEPS="${STEPS:-}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-}"
LR="${LR:-}"
HPARAMS_PROFILE="${HPARAMS_PROFILE:-paper}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-m2cl-domain-generalization}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"

COMMON_ARGS=(
  --data_root "${DATA_ROOT}"
  --methods "${METHOD_ARGS[@]}"
  --backbones "${BACKBONE}"
  --seeds "${SEED_ARGS[@]}"
  --hparams_profile "${HPARAMS_PROFILE}"
  --num_workers "${NUM_WORKERS}"
  --save_dir "${SAVE_DIR}"
  --skip_existing
)

if [[ -n "${EPOCHS}" ]]; then
  COMMON_ARGS+=(--epochs "${EPOCHS}")
fi

if [[ -n "${STEPS}" ]]; then
  COMMON_ARGS+=(--steps "${STEPS}")
fi

if [[ -n "${CHECKPOINT_FREQ}" ]]; then
  COMMON_ARGS+=(--checkpoint_freq "${CHECKPOINT_FREQ}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  COMMON_ARGS+=(--batch_size "${BATCH_SIZE}")
fi

if [[ -n "${LR}" ]]; then
  COMMON_ARGS+=(--lr "${LR}")
fi

case "${WANDB}" in
  1|true|TRUE|yes|YES)
    COMMON_ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY}" ]]; then
      COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
    fi
    if [[ -n "${WANDB_GROUP}" ]]; then
      COMMON_ARGS+=(--wandb_group "${WANDB_GROUP}")
    fi
    if [[ -n "${WANDB_MODE}" ]]; then
      COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
    fi
    if [[ -n "${WANDB_TAGS}" ]]; then
      read -r -a WANDB_TAG_ARGS <<< "${WANDB_TAGS}"
      COMMON_ARGS+=(--wandb_tags "${WANDB_TAG_ARGS[@]}")
    fi
    ;;
esac

for dataset in "${DATASET_ARGS[@]}"; do
  echo "Running ${dataset}: methods=${METHODS}, seeds=${SEEDS}, backbone=${BACKBONE}"
  python run_experiments.py --dataset "${dataset}" "${COMMON_ARGS[@]}"
done

python summarize_results.py \
  --metrics_dir "${SAVE_DIR}" \
  --output_csv "outputs/main_${BACKBONE}_results.csv"

echo "Main experiments finished."
echo "Checkpoints: ${SAVE_DIR}"
echo "Summary CSV: outputs/main_${BACKBONE}_results.csv"
