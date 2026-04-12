#!/bin/bash
# launch_eval.sh - Launch N parallel server+client pairs for faster evaluation.

set -euo pipefail

# ---- Conda init for tmux ----
CONDA_BASE="$(conda info --base)"
CONDA_INIT="source $CONDA_BASE/etc/profile.d/conda.sh"

# ---- Defaults ----
NUM_INSTANCES=2
GPUS="0,1"
BASE_PORT=7000
BENCHMARK_DIR=""
CHECKPOINT_PATH=""
OUTPUT_DIR="./eval_output"
SERVER_CONDA_ENV="dreamzero"
CLIENT_CONDA_ENV="mlspaces"
DREAMZERO_DIR="/home/jianzhang/zdj/dreamzero"
MOLMOSPACES_DIR="/home/jianzhang/zdj/molmospaces"
ASSETS_DIR="/home/jianzhang/zdj/molmospaces_assets"
IMAGE_HEIGHT=180
SERVE_OUTPUT=""
EXTRA_SERVER_ARGS=""
EXTRA_CLIENT_ARGS=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-instances) NUM_INSTANCES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --assets-dir) ASSETS_DIR="$2"; shift 2 ;;
        --image-height) IMAGE_HEIGHT="$2"; shift 2 ;;
        --serve-output) SERVE_OUTPUT="$2"; shift 2 ;;
        --server-conda-env) SERVER_CONDA_ENV="$2"; shift 2 ;;
        --client-conda-env) CLIENT_CONDA_ENV="$2"; shift 2 ;;
        --wandb) EXTRA_CLIENT_ARGS="$EXTRA_CLIENT_ARGS --wandb"; shift ;;
        --no-save-data) EXTRA_CLIENT_ARGS="$EXTRA_CLIENT_ARGS --no-save-data"; shift ;;
        --no-save-video) EXTRA_CLIENT_ARGS="$EXTRA_CLIENT_ARGS --no-save-video"; shift ;;
        --recent-ref-only) EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --recent-ref-only"; shift ;;
        --enable-video-token-pruning) EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --enable-video-token-pruning"; shift ;;
        --pruning-score-layer) EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --pruning-score-layer $2"; shift 2 ;;
        --pruning-schedule) EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --pruning-schedule $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$BENCHMARK_DIR" || -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --benchmark-dir and --checkpoint-path are required."
    exit 1
fi

echo "=== Parallel Evaluation ==="
echo "  Instances:      $NUM_INSTANCES"
echo "  GPUs:           $GPUS"
echo "  Ports:          $BASE_PORT - $((BASE_PORT + NUM_INSTANCES - 1))"
echo "  Benchmark:      $BENCHMARK_DIR"
echo "  Checkpoint:     $CHECKPOINT_PATH"
echo "  Output:         $OUTPUT_DIR"
echo "  Serve output:   ${SERVE_OUTPUT:-<cwd>}"
echo "  Image height:   $IMAGE_HEIGHT (14B=180, 5B=160)"
echo ""

# Build serve-output args for server
SERVE_OUTPUT_ARGS=""
if [[ -n "$SERVE_OUTPUT" ]]; then
    mkdir -p "$SERVE_OUTPUT"
    SERVE_OUTPUT_ARGS="--serve-output $SERVE_OUTPUT"
fi

# ---- Start servers one by one (wait for each to be ready before starting next) ----
echo "Starting $NUM_INSTANCES server instances (sequentially to avoid GPU conflicts)..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    SESSION="server_${i}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    SERVER_LOG="${SERVE_OUTPUT:-$DREAMZERO_DIR}/server_${i}.log"
    tmux new -d -s "$SESSION" "bash -c '$CONDA_INIT && conda activate $SERVER_CONDA_ENV && cd $DREAMZERO_DIR && CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=$((29500 + i)) socket_test_optimized_AR.py --port $PORT --enable-dit-cache --model-path $CHECKPOINT_PATH --no-save-video --image-height $IMAGE_HEIGHT $SERVE_OUTPUT_ARGS $EXTRA_SERVER_ARGS 2>&1 | tee $SERVER_LOG; exec bash'"
    echo -n "  [server_$i] port=$PORT — waiting for ready..."

    RETRIES=0
    MAX_RETRIES=600  # 10 minutes per server
    while ! curl -s "http://localhost:$PORT/healthz" > /dev/null 2>&1; do
        sleep 1
        RETRIES=$((RETRIES + 1))
        if [ $RETRIES -ge $MAX_RETRIES ]; then
            echo " TIMEOUT (${MAX_RETRIES}s). Check: tmux attach -t server_$i"
            exit 1
        fi
    done
    echo " ready!"
done

# ---- Start clients ----
echo ""
echo "Starting $NUM_INSTANCES client shards..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    SESSION="client_${i}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new -d -s "$SESSION" "bash -c '$CONDA_INIT && conda activate $CLIENT_CONDA_ENV && cd $MOLMOSPACES_DIR && xvfb-run -a python run_molmo_spaces_eval.py --benchmark-dir $BENCHMARK_DIR --checkpoint-path $CHECKPOINT_PATH --host localhost --port $PORT --image-height $IMAGE_HEIGHT --num-shards $NUM_INSTANCES --shard-id $i --output-dir $OUTPUT_DIR --assets-dir $ASSETS_DIR $EXTRA_CLIENT_ARGS 2>&1 | tee $MOLMOSPACES_DIR/client_${i}.log; exec bash'"
    echo "  [client_$i] shard=$i/$NUM_INSTANCES port=$PORT tmux=$SESSION"
done

echo ""
echo "=== All instances launched ==="
echo ""
echo "Monitor:"
echo "  tmux attach -t server_0   # view server 0"
echo "  tmux attach -t client_0   # view client 0"
echo "  nvidia-smi                # check GPU usage"
echo ""
echo "Stop all:"
echo "  for i in \$(seq 0 $((NUM_INSTANCES - 1))); do tmux kill-session -t server_\$i; tmux kill-session -t client_\$i; done"
# cd /home/jianzhang/zdj/dreamzero
#   bash launch_eval.sh \
#     --num-instances 3 \
#     --gpus 0,1 \
#     --benchmark-dir
#   /home/jianzhang/zdj/molmospaces_assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231 \
#     --checkpoint-path /home/jianzhang/zdj/Droid_5B_18k \
#     --wandb --no-save-data
# --image-height 180 
# --recent-ref-only
# --serve-output /home/jianzhang/zdj/server_output
# --enable-video-token-pruning

#   tmux kill-session -t server_0; tmux kill-session -t server_1; tmux kill-session -t server_2
#   tmux kill-session -t client_0; tmux kill-session -t client_1; tmux kill-session -t client_2



### prune
#   cd /home/jianzhang/zdj/dreamzero

#   bash launch_eval.sh \
#     --num-instances 3 \
#     --gpus 0,1 \
#     --benchmark-dir /path/to/your/benchmark \
#     --checkpoint-path /path/to/your/checkpoint \
#     --enable-video-token-pruning \
#     --wandb --no-save-data

#   默认参数:
#   - pruning_score_layer = 20（第20层计算 importance score）
#   - pruning_schedule = [1.0, 1.0, 0.5, ..., 0.25, 0.25, 0.25]（16步，渐进式）

#   如果想调参:
#   # 更激进的剪枝（全程 keep 25%）
#   --pruning-schedule "1.0,1.0,0.25,1.0,1.0,1.0,0.25,1.0,1.0,1.0,0.25,1.0,1.0,0.25,0.25,0.25"

#   # 更保守的剪枝（全程 keep 50%）
#   --pruning-schedule "1.0,1.0,0.5,1.0,1.0,1.0,0.5,1.0,1.0,1.0,0.5,1.0,1.0,0.5,0.5,0.5"

#   # 换 scoring layer
#   --pruning-score-layer 15

#   对比基线: 不加 --enable-video-token-pruning 跑一组，对比 success rate 和 timing。