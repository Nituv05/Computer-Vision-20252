#!/usr/bin/env bash
# =============================================================================
# scripts/sync-train.sh  —  Đẩy code LAPTOP -> SERVER, chọn 1 H100, train song song, báo xong.
#
#   Chạy TRÊN LAPTOP (Kiro/đầu não ở đây). Server chỉ là "máy cày".
#
#   CÁCH DÙNG (mỗi lệnh con là 1 việc):
#     # --- 1 LỆNH LÀM HẾT từ laptop: setup nếu cần + data nếu thiếu + train ---
#     cd <thư-mục-project> && bash scripts/sync-train.sh
#     # --- hoặc tách từng bước nếu muốn debug ---
#     bash scripts/sync-train.sh setup               # đẩy code + cài Miniconda + env + deps
#     bash scripts/sync-train.sh data                # tải dataset (pacs/vlcs/office_home) về server
#     bash scripts/sync-train.sh run                 # chỉ đẩy code + bật train
#     bash scripts/sync-train.sh log                 # xem log realtime (lần chạy mới nhất)
#     bash scripts/sync-train.sh logs                # liệt kê TẤT CẢ log các lần train trước
#     bash scripts/sync-train.sh attach              # vào màn hình job (thoát: Ctrl-b d)
#     bash scripts/sync-train.sh status              # job còn chạy? GPU thế nào?
#     bash scripts/sync-train.sh pull                # kéo logs+outputs từ server VỀ LAPTOP (backup)
#     bash scripts/sync-train.sh watch               # đợi train xong: vừa chờ vừa kéo log về (notify)
#     bash scripts/sync-train.sh stop                # dừng job
#     # W&B: mở .env.wandb trên laptop, dán WANDB_API_KEY rồi chạy scripts/sync-train.sh
# =============================================================================

set -euo pipefail   # -e: lỗi là DỪNG | -u: dùng biến chưa khai báo -> báo lỗi
                    # -o pipefail: lỗi giữa pipe cũng bị bắt (không bị nuốt âm thầm)

# ============================== CONFIG =======================================
# Chỉ sửa phần này cho mỗi project. Phần dưới CONFIG không cần đụng.

SRC="$(pwd)/"                 # Code trên LAPTOP. Mặc định = thư mục hiện tại.
                              # Dấu "/" cuối BẮT BUỘC: rsync = đẩy NỘI DUNG thư mục này.

HOST="viettel_ai"          # Alias server trong ~/.ssh/config.

DEST="/home/nvidia-lab/ai4life/annd/respiratory"         # Thư mục đích TRÊN SERVER viettel_ai.

ENV="res"                    # Tên conda env trên server (tạo sẵn ở bước "làm 1 lần").

SESSION="res1"                # Tên tmux session. RIÊNG để người khác không kill nhầm.

USE_WANDB=1                   # 1 = bật W&B khi train; 0 = tắt. Key lấy từ file $WANDB_ENV_FILE.
WANDB_ENV_FILE=".env.wandb"   # File secret local. Bị gitignore, nhưng scripts/sync-train.sh sẽ đẩy riêng lên server.

ENTRY="SEEDS=0 BATCH_SIZE=32 bash scripts/run_paper.sh"
                              # Chạy ResNet-18, chỉ seed 0. BATCH_SIZE=16 để 2 dataset song song
                              # giữ VRAM dưới mục tiêu khoảng 25GB.

DATA_ROOT="data_root"         # Thư mục dataset TRÊN SERVER (đặt trong $DEST). Lệnh 'data' tải vào đây.
DATASETS="vlcs office_home"   # Chạy 2 dataset này song song trên cùng H100.

LOCAL_BACKUP="$(pwd)/server-backup"   # Nơi lưu logs + outputs nhẹ kéo từ server VỀ LAPTOP; không kéo checkpoint.

PYVER="3.11"                  # Python cho conda env (runbook yêu cầu 3.10/3.11).

# --- Tính năng nâng cấp: bật/tắt + tham số ---
TARGET_GPU="auto"             # auto = tự chọn H100 trống nhất. Có thể đặt số GPU cụ thể, ví dụ TARGET_GPU=0.
TARGET_GPU_NAME_MATCH="H100"  # Chỉ dùng GPU có tên chứa chuỗi này, tránh lấy nhầm GPU khác.
TRAIN_VRAM_BUDGET_MB=25000    # Mục tiêu: phần train của mình dùng dưới khoảng 25GB VRAM.
MIN_FREE_MB="$TRAIN_VRAM_BUDGET_MB"
                              # Chỉ cần GPU còn >= 25GB free để chạy 2 job ResNet-18 seed0 song song.

RESUME=0                      # 1 = tự tìm checkpoint mới nhất để train tiếp; 0 = train từ đầu.
CKPT_DIR="$DEST/outputs"      # Nơi server lưu checkpoint (đường TRÊN SERVER).
CKPT_GLOB="*.pth"             # Mẫu file checkpoint (đổi thành *.ckpt nếu dùng Lightning...).
RESUME_FLAG="--resume"        # Cờ mà tools/train.py NHẬN để nạp checkpoint. << tools/train.py phải hỗ trợ cờ này.

CONDA_HOME="/home/nvidia-lab/miniconda3"
CONDA_SH="$CONDA_HOME/etc/profile.d/conda.sh"
CUDA_HOME_REMOTE="auto"       # auto = dùng /usr/local/cuda nếu có, fallback theo nvcc. Có thể set path cụ thể.
                              # SSH non-interactive cần source đúng conda.sh, không dựa vào .bashrc.
# =============================================================================

# Guard: bắt buộc đổi DEST, tránh đẩy nhầm vào thư mục placeholder.
[[ "$DEST" == *CHANGE_ME* ]] && { echo "❌ Hãy sửa DEST trong CONFIG trước."; exit 1; }

ACTION="${1:-all}"            # Mặc định = all: setup nếu cần + data nếu thiếu + train.
CODE_PUSHED=0

wandb_enabled() {
  case "$USE_WANDB" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

wandb_env_ready() {
  local file="${SRC}${WANDB_ENV_FILE}"
  [[ -f "$file" ]] || return 1
  (
    set +u
    set -a
    source "$file"
    set +a
    [[ -n "${WANDB_API_KEY:-}" && "$WANDB_API_KEY" != "PASTE_YOUR_WANDB_API_KEY_HERE" ]]
  )
}

require_wandb_env() {
  wandb_enabled || return 0
  if ! wandb_env_ready; then
    echo "❌ W&B đang bật (USE_WANDB=$USE_WANDB) nhưng $WANDB_ENV_FILE chưa có WANDB_API_KEY."
    echo "   Mở $WANDB_ENV_FILE trên laptop và điền:"
    echo '   WANDB_API_KEY="PASTE_YOUR_WANDB_API_KEY_HERE"'
    echo "   Nếu muốn train không dùng W&B, đặt USE_WANDB=0 trong scripts/sync-train.sh."
    exit 1
  fi
}

push_wandb_env() {
  local file="${SRC}${WANDB_ENV_FILE}"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  if ! wandb_env_ready; then
    echo "⚠  $WANDB_ENV_FILE chưa có WANDB_API_KEY hợp lệ, bỏ qua đẩy file W&B env."
    return 0
  fi
  echo "==> Đẩy W&B env: $WANDB_ENV_FILE -> $HOST:$DEST/$WANDB_ENV_FILE"
  rsync -avz "$file" "$HOST:$DEST/$WANDB_ENV_FILE"
}

# ----- Helper: ĐẨY CODE laptop -> server (dùng lại cho run + setup) -----
#   --delete: server khớp Y HỆT laptop. --exclude bảo vệ data/output/log server tự sinh.
#   LƯU Ý repo này: 'data/' là MÃ NGUỒN (package dataset), KHÔNG phải dataset -> KHÔNG exclude.
#   Dataset thật nằm ở '$DATA_ROOT/' (vd data_root/) -> exclude cái đó.
#   Các .env* khác bị exclude để tránh đẩy nhầm secret; riêng $WANDB_ENV_FILE được đẩy bằng push_wandb_env().
push_code() {
  echo "==> Đẩy code: $SRC -> $HOST:$DEST/"
  rsync -avz --delete \
    --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
    --exclude '.env' --exclude '.env.*' \
    --exclude "$DATA_ROOT/" --exclude 'data_root/' --exclude 'datasets/' \
    --exclude 'outputs/' --exclude 'runs/' --exclude 'wandb/' --exclude 'server-backup/' \
    --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' --exclude 'logs/' \
    "$SRC" "$HOST:$DEST/"
  push_wandb_env
  CODE_PUSHED=1
}

# ----- Helper: KÉO logs + outputs nhẹ server -> laptop; KHÔNG kéo checkpoint/weights -----
pull_results() {
  mkdir -p "$LOCAL_BACKUP"
  rsync -avz "$HOST:$DEST/logs/" "$LOCAL_BACKUP/logs/" 2>/dev/null || true
  rsync -avz \
    --exclude '*.pth*' --exclude '*.pt*' --exclude '*.ckpt*' --exclude '*.safetensors*' \
    "$HOST:$DEST/outputs/" "$LOCAL_BACKUP/outputs/" 2>/dev/null || true
  rsync -avz "$HOST:$DEST/STATUS" "$LOCAL_BACKUP/STATUS" 2>/dev/null || true
}

# ----- Helper: cài Miniconda (nếu thiếu) + tạo conda env + cài deps TRÊN SERVER -----
do_setup() {
  push_code
  echo "==> Cài môi trường trên $HOST (Miniconda + conda env '$ENV' + deps) ..."
  ssh "$HOST" "bash -s" <<EOF
set -euo pipefail
if [ ! -f "$CONDA_SH" ]; then
  echo '-> Chưa có conda, tải Miniconda...'
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_HOME"
  rm -f /tmp/mc.sh
fi
source "$CONDA_SH"
conda env list | grep -qE "^$ENV\s" || conda create -y -n "$ENV" python=$PYVER
conda activate "$ENV"
if [ "$CUDA_HOME_REMOTE" = "auto" ]; then
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="\$(dirname "\$(dirname "\$(command -v nvcc)")")"
  fi
elif [ -n "$CUDA_HOME_REMOTE" ]; then
  export CUDA_HOME="$CUDA_HOME_REMOTE"
fi
if [ -n "\${CUDA_HOME:-}" ]; then
  export PATH="\$CUDA_HOME/bin:\$PATH"
  export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
fi
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$DEST/requirements.txt"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
EOF
  echo "✅ Setup xong."
}

# ----- Helper: tải dataset về server (vào \$DEST/\$DATA_ROOT). -----
do_data() {
  echo "==> Tải dataset [$DATASETS] vào $HOST:$DEST/$DATA_ROOT ..."
  ssh "$HOST" "bash -s" <<EOF
set -euo pipefail
source "$CONDA_SH"
conda activate "$ENV"
if [ "$CUDA_HOME_REMOTE" = "auto" ]; then
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="\$(dirname "\$(dirname "\$(command -v nvcc)")")"
  fi
elif [ -n "$CUDA_HOME_REMOTE" ]; then
  export CUDA_HOME="$CUDA_HOME_REMOTE"
fi
if [ -n "\${CUDA_HOME:-}" ]; then
  export PATH="\$CUDA_HOME/bin:\$PATH"
  export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
fi
cd "$DEST"
python tools/download_data.py --data_root "$DATA_ROOT" --datasets $DATASETS
for DS in $DATASETS; do
  python tools/check_data.py --data_root "$DATA_ROOT" --dataset "$DS"
done
EOF
}

if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
  require_wandb_env
fi

# ----- Các lệnh con NGẮN: làm xong thoát ngay -----
case "$ACTION" in
  all)
    # 1 LỆNH TỪ LAPTOP làm HẾT: setup môi trường -> tải data nếu thiếu -> rơi xuống phần 'run' để train.
    do_setup
    DATA_READY=1
    for DS in $DATASETS; do
      if ! ssh "$HOST" "[ -d '$DEST/$DATA_ROOT/'"$DS"' ]"; then
        DATA_READY=0
      fi
    done
    if [[ "$DATA_READY" == "1" ]]; then
      echo "==> Dataset đã có [$DATASETS], bỏ qua tải."
    else
      do_data
    fi
    ;;                          # KHÔNG exit -> chạy tiếp xuống ACTION=run bên dưới để train luôn.
  setup)  do_setup; echo "   Tải dataset: bash scripts/sync-train.sh data   |   Hoặc làm hết 1 lệnh: bash scripts/sync-train.sh all"; exit 0 ;;
  data)   do_data; exit 0 ;;
  pull)   pull_results; echo "✅ Đã kéo logs+outputs nhẹ (không checkpoint) về: $LOCAL_BACKUP"; exit 0 ;;
  log)    exec ssh "$HOST" "cd $DEST && files=\$(ls -t logs/*.log 2>/dev/null | head -n 6); [ -n \"\$files\" ] && tail -n 80 -f \$files || echo '(chưa có log)'" ;;
  logs)   exec ssh "$HOST" "ls -lht $DEST/logs/*.log 2>/dev/null || echo '(chưa có log)'" ;;
  attach) exec ssh "$HOST" -t "tmux attach -t $SESSION" ;;                  # vào màn hình job
  stop)   exec ssh "$HOST" "tmux kill-session -t $SESSION" ;;               # dừng job
  status)
    # Job còn sống không? + nội dung file STATUS (do job ghi khi kết thúc) + GPU.
    ssh "$HOST" "tmux has-session -t $SESSION 2>/dev/null \
      && echo '● ĐANG CHẠY (session $SESSION)' \
      || echo '○ KHÔNG có session $SESSION (đã xong hoặc chưa chạy)'; \
      echo -n 'STATUS: '; cat $DEST/STATUS 2>/dev/null || echo '(chưa có)'; \
      echo '--- GPU ---'; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader"
    exit 0 ;;
  watch)
    # Vừa ĐỢI job xong vừa KÉO logs+outputs nhẹ về laptop định kỳ; không kéo checkpoint.
    echo "⏳ Đợi job '$SESSION' xong, vừa chờ vừa backup về $LOCAL_BACKUP ... (Ctrl-c để thôi đợi, job vẫn chạy)"
    while ssh "$HOST" "tmux has-session -t $SESSION 2>/dev/null"; do
      pull_results            # backup định kỳ trong lúc train
      sleep 30
    done
    pull_results              # kéo lần cuối sau khi xong
    RESULT="$(ssh "$HOST" "cat $DEST/STATUS 2>/dev/null" || true)"
    MSG="Train [$HOST:$SESSION] xong — ${RESULT:-?}"
    command -v notify-send >/dev/null && notify-send "Kiro train" "$MSG"     # popup nếu có
    printf '\a'; echo "🔔 $MSG  (backup tại $LOCAL_BACKUP)"                   # chuông + in ra
    exit 0 ;;
esac

# ============================ ACTION = run ===================================
# Chọn 1 H100 80GB và chạy 2 dataset song song trên cùng GPU đó.
if [[ "$CODE_PUSHED" != "1" ]]; then
  push_code
fi

# Không bật chồng job (tránh đua GPU).
if ssh "$HOST" "tmux has-session -t $SESSION 2>/dev/null"; then
  echo "⚠  Session '$SESSION' đang chạy. Dừng nó trước: bash scripts/sync-train.sh stop"; exit 1
fi

read -r -a DATASET_ARR <<< "$DATASETS"
if (( ${#DATASET_ARR[@]} != 2 )); then
  echo "⚠  DATASETS hiện có ${#DATASET_ARR[@]} dataset: $DATASETS"
  echo "   Script vẫn chạy song song tất cả dataset trên 1 GPU, nhưng cấu hình VRAM đang tính cho 2 dataset."
fi

GPU_QUERY="nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits"
if [[ "$TARGET_GPU" == "auto" ]]; then
  SELECTED_GPU="$(ssh "$HOST" "$GPU_QUERY" | awk -F, -v pat="$TARGET_GPU_NAME_MATCH" -v min="$MIN_FREE_MB" '
    {
      idx=$1; name=$2; free=$3+0; total=$4+0;
      gsub(/^ +| +$/, "", idx); gsub(/^ +| +$/, "", name);
      if (name ~ pat && free >= min && free > best_free) {
        best_idx=idx; best_name=name; best_free=free; best_total=total;
      }
    }
    END { if (best_idx != "") print best_idx "|" best_name "|" best_free "|" best_total; }
  ')"
else
  SELECTED_GPU="$(ssh "$HOST" "$GPU_QUERY" | awk -F, -v target="$TARGET_GPU" -v pat="$TARGET_GPU_NAME_MATCH" -v min="$MIN_FREE_MB" '
    {
      idx=$1; name=$2; free=$3+0; total=$4+0;
      gsub(/^ +| +$/, "", idx); gsub(/^ +| +$/, "", name);
      if (idx == target && name ~ pat && free >= min) print idx "|" name "|" free "|" total;
    }
  ')"
fi

if [[ -z "$SELECTED_GPU" ]]; then
  echo "❌ Không tìm thấy GPU '$TARGET_GPU_NAME_MATCH' đủ trống."
  echo "   Cần free >= ${MIN_FREE_MB}MB cho train budget khoảng ${TRAIN_VRAM_BUDGET_MB}MB."
  echo "   GPU hiện tại:"
  ssh "$HOST" "$GPU_QUERY" || true
  exit 1
fi

IFS='|' read -r RUN_GPU RUN_GPU_NAME RUN_GPU_FREE RUN_GPU_TOTAL <<< "$SELECTED_GPU"
echo "==> Chọn GPU $RUN_GPU: $RUN_GPU_NAME, free ${RUN_GPU_FREE}MB / total ${RUN_GPU_TOTAL}MB"
echo "==> Chạy ${#DATASET_ARR[@]} dataset song song trên cùng GPU, train budget mục tiêu ~${TRAIN_VRAM_BUDGET_MB}MB."

LAUNCH=""
for DS in "${DATASET_ARR[@]}"; do
  echo "    -> GPU $RUN_GPU : $DS"
  LAUNCH+="( CUDA_VISIBLE_DEVICES=$RUN_GPU DATASETS=$DS SKIP_SUMMARY=1 $ENTRY \
    > logs/${DS}_\$TS.log 2>&1; echo \"\$? $DS\" >> logs/exit_\$TS.txt ) &"$'\n'
done

ssh "$HOST" "cat > $DEST/.run.sh" <<EOF
set -euo pipefail
source "$CONDA_SH"
conda activate "$ENV"
if [ "$CUDA_HOME_REMOTE" = "auto" ]; then
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="\$(dirname "\$(dirname "\$(command -v nvcc)")")"
  fi
elif [ -n "$CUDA_HOME_REMOTE" ]; then
  export CUDA_HOME="$CUDA_HOME_REMOTE"
fi
if [ -n "\${CUDA_HOME:-}" ]; then
  export PATH="\$CUDA_HOME/bin:\$PATH"
  export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
fi
cd "$DEST"
export WANDB="$USE_WANDB"
if [ -f "$WANDB_ENV_FILE" ]; then
  set -a
  source "$WANDB_ENV_FILE" || exit 1
  set +a
fi
case "\${WANDB:-0}" in
  1|true|TRUE|yes|YES)
    if [ "\${WANDB_MODE:-online}" != "offline" ] && [ -z "\${WANDB_API_KEY:-}" ]; then
      echo "Missing WANDB_API_KEY in $WANDB_ENV_FILE. Set WANDB_MODE=offline or add the key." >&2
      exit 1
    fi
    if [ -n "\${WANDB_API_KEY:-}" ]; then
      wandb login --relogin "\$WANDB_API_KEY" >/dev/null || exit 1
    fi
    ;;
esac
mkdir -p logs outputs
TS=\$(date '+%Y%m%d_%H%M%S')
ln -sfn "${DATASET_ARR[0]}_\$TS.log" logs/latest.log    # 'log' bám dataset đầu; xem cái khác: bash scripts/sync-train.sh logs
echo RUNNING > STATUS
set +e                                                 # các job nền tự ghi exit code; không dừng job khác khi 1 job fail
$LAUNCH
wait                                                    # đợi TẤT CẢ nhóm GPU chạy xong
set -e
# Summarize 1 lần sau khi mọi job xong.
python tools/summarize_results.py --metrics_dir outputs/paper30_r18 \
  --output_csv outputs/paper30_r18_results.csv 2>&1 | tee -a logs/summary_\$TS.log || true
echo "EXIT=\$(cat logs/exit_\$TS.txt 2>/dev/null | tr '\n' ';') \$(date '+%F %T')" > STATUS
EOF
ssh "$HOST" "tmux new-session -d -s $SESSION 'bash $DEST/.run.sh'"

echo "✅ Đã bật training (${#DATASET_ARR[@]} dataset song song trên GPU $RUN_GPU: $RUN_GPU_NAME)."
echo "   Xem log  : bash scripts/sync-train.sh logs   (mỗi dataset 1 file <dataset>_<time>.log)"
echo "   Backup   : bash scripts/sync-train.sh pull   (kéo logs+outputs nhẹ, không checkpoint, về $LOCAL_BACKUP)"
echo "   Đợi xong : bash scripts/sync-train.sh watch  (vừa đợi vừa tự backup về laptop)"
echo "   Tình trạng: bash scripts/sync-train.sh status   |   Dừng: bash scripts/sync-train.sh stop"
