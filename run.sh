#!/usr/bin/env bash
# Run ConViT on CIFAR-10 (multi-GPU) with the legacy environment.

set -euo pipefail

# ----------------- CONFIG -----------------
ENV_NAME=${ENV_NAME:-convit_legacy}     # conda env with torch==1.8.1 timm==0.3.2
DATA_PATH=${DATA_PATH:-"./data"}        # CIFAR-10 dataset root
OUTPUT_DIR=${OUTPUT_DIR:-"exp/c10/baseline_convit_base"}
MODEL=${MODEL:-"convit_base"}
BATCH_SIZE=${BATCH_SIZE:-64}            # per-GPU batch size
EPOCHS=${EPOCHS:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
MASTER_PORT=${MASTER_PORT:-29571}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4"}   # set GPUs to use

# ----------------- ENV PYTHON -----------------
CONDA_BASE="$(conda info --base)"
ENV_PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "$ENV_PY" ]]; then
  echo "ERROR: cannot find python for env '$ENV_NAME' at $ENV_PY"
  exit 1
fi

# ----------------- SANITY -----------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}
mkdir -p "$OUTPUT_DIR"

echo "=== ConViT CIFAR-10 run ==="
echo "Python:       $ENV_PY"
echo "Env:          $ENV_NAME"
echo "GPUs:         $NPROC  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Model:        $MODEL"
echo "Data path:    $DATA_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "Batch/GPU:    $BATCH_SIZE"
echo "Epochs:       $EPOCHS"
echo "Num workers:  $NUM_WORKERS"
echo "Master port:  $MASTER_PORT"
echo "================================="

# Clean stale caches
find . -name "__pycache__" -type d -exec rm -rf {} + || true
find . -name "*.pyc" -delete || true

# Print versions from env
"$ENV_PY" - <<'PY'
import sys, torch, timm
print("Python:", sys.version.split()[0])
print("torch :", torch.__version__, "CUDA OK:", torch.cuda.is_available())
print("timm  :", timm.__version__)
PY

# ----------------- LAUNCH -----------------
# NOTE: --use_env prevents --local_rank from being passed as a CLI arg.
#       Your script should read LOCAL_RANK / RANK / WORLD_SIZE from env.
"$ENV_PY" -m torch.distributed.launch \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  --use_env \
  main.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --data-path "$DATA_PATH" \
    --data-set CIFAR10 \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS"
