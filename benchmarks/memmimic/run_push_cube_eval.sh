#!/usr/bin/env bash
# MemMimic push_cube rollout of an OpenWAM / Memory-OpenWAM checkpoint.
#
# Usage:
#   bash benchmarks/memmimic/run_push_cube_eval.sh <ckpt_dir> <run_name> [episode_num=100] [env_num=6] [hydra overrides...]
#
# Serves the checkpoint with benchmarks/memmimic/robotmq_policy_server.py (OpenWAM venv,
# PPU/CUDA device $DEVICE) and runs GMP's mujoco-env parallel rollout against it
# (conda env `mujoco-env`, OSMesa software rendering unless MUJOCO_GL is set).
# Outputs: $OUT_ROOT/<run_name>/{policy_server.log,rollout.log,rollout/}.
set -uo pipefail
CKPT_DIR=$1; RUN=$2; EPISODES=${3:-100}; ENV_NUM=${4:-6}; shift 4 2>/dev/null || shift $#

OPENWAM=${OPENWAM:-/mnt/cpfs/workspace/OpenWAM}
GMP_ROOT=${GMP_ROOT:-/mnt/cpfs/workspace/RSS/gated-memory-policy}
MJ_PY=${MJ_PY:-/mnt/cpfs/workspace/tools/miniforge3/envs/mujoco-env/bin/python}
OUT_ROOT=${OUT_ROOT:-/mnt/cpfs/workspace/memory-openwam/outputs/eval}
PORT=${PORT:-18765}
DEVICE=${DEVICE:-cuda:0}
CKPT_NAME=${CKPT_NAME:-}
START_SEED=${START_SEED:-10005}
# Memory inference path (memory-openwam/docs/07). Default "auto" = the streaming KV path whenever the
# checkpoint allows it (memory on, no action history, no eval-time ablation), else the original
# recompute path. Anything in SERVER_ARGS comes later on the command line and overrides it.
MEMORY_INFERENCE=${MEMORY_INFERENCE:-auto}
OUT=$OUT_ROOT/$RUN
mkdir -p "$OUT"

# One rollout at a time on this host (single PPU, fixed port): later callers queue here.
exec 9>/tmp/memory_openwam_push_cube_eval.lock
echo "[eval] waiting for the eval lock ($(date -u +%FT%TZ))"
flock 9
echo "[eval] lock acquired ($(date -u +%FT%TZ))"

# The conda envs hardcode the pre-migration CPFS prefix (see RSS/RESTORE_GMP_20260919.md).
[ -e /mnt/cpfs/PeterX ] || ln -s /mnt/cpfs/workspace /mnt/cpfs/PeterX

echo "[eval] ckpt=$CKPT_DIR run=$RUN episodes=$EPISODES env_num=$ENV_NUM out=$OUT"

# --- policy server (OpenWAM venv) ---
(
  cd "$OPENWAM" || exit 1
  source scripts/env.sh >/dev/null 2>&1
  export TMPDIR=/tmp
  exec .venv/bin/python benchmarks/memmimic/robotmq_policy_server.py \
    --ckpt-dir "$CKPT_DIR" ${CKPT_NAME:+--ckpt-name "$CKPT_NAME"} \
    --device "$DEVICE" --endpoint "tcp://0.0.0.0:$PORT" \
    --memory-inference "$MEMORY_INFERENCE" --timing-jsonl "$OUT/timing.jsonl" ${SERVER_ARGS:-}
) > "$OUT/policy_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q "Waiting for environment requests" "$OUT/policy_server.log"; do
  kill -0 $SERVER 2>/dev/null || { echo "[eval] policy server died; see $OUT/policy_server.log"; tail -30 "$OUT/policy_server.log"; exit 1; }
  sleep 5
done
echo "[eval] policy server up (pid $SERVER)"

# --- simulator rollout (GMP mujoco-env) ---
cd "$GMP_ROOT/mujoco-env" || exit 1
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export POLICY_INFERENCE_TIMEOUT_S=${POLICY_INFERENCE_TIMEOUT_S:-1800}
export PYTHONUNBUFFERED=1 TMPDIR=/tmp
"$MJ_PY" scripts/rollout_policy_parallel.py \
  policy_server_port=$PORT task_name=push_cube \
  task.env_num=$ENV_NUM episode_num=$EPISODES start_seed=$START_SEED \
  task.data_storage_dir="$OUT/rollout" \
  +task.agent.action_prediction_horizon=32 +task.agent.action_execution_horizon=16 \
  '+task.render_image_indices=[-13,-9,-5,-1]' \
  "$@" > "$OUT/rollout.log" 2>&1
RC=$?
echo "[eval] rollout exit $RC"
grep -E "Success rate|Mean reward|Time taken" "$OUT/rollout.log" | tail -3
