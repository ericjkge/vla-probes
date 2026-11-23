#!/usr/bin/env bash
# pi0.5 (Physical Intelligence's pi05_libero) runs the 5 demo instructions in their harmful and harmless scenes; videos + results.json.
# Run from the repo root on the cluster:   sbatch libero_safety_suites/demo_pi05.sbatch
# Needs the one-time setup under $VLA_HARM_ROOT (env/, LIBERO-Safety/, openpi/): bash Old/cluster/setup.sh && bash Old/cluster/setup_pi05.sh
# Extra arguments go to run_demo_pi05.py, e.g.   sbatch libero_safety_suites/demo_pi05.sbatch --demo 1,4
#SBATCH --job-name=demo-pi05
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
module load ncf/1.0.0-fasrc01 2>/dev/null || true
module load miniconda3/py39_4.11.0-linux_x64-ncf 2>/dev/null || true
export VLA_HARM_ROOT="${VLA_HARM_ROOT:-$HOME/vla-harm}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY="$VLA_HARM_ROOT/env/bin/python"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "node $(hostname)  job ${SLURM_JOB_ID:-}  gpus ${CUDA_VISIBLE_DEVICES:-none}"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

# the model runs as a local policy server in openpi's own environment; the simulator script connects to it
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000)); SLOG="logs/pi05-server-${SLURM_JOB_ID:-0}.log"
export OPENPI_DATA_HOME="$VLA_HARM_ROOT/openpi_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false
( cd "$VLA_HARM_ROOT/openpi" && exec .venv/bin/python scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config=pi05_libero --policy.dir="${PI05_CKPT:-gs://openpi-assets/checkpoints/pi05_libero}" ) > "$SLOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
echo "pi05 server pid $SERVER_PID port $PORT, log $SLOG; waiting for it to load..."
UP=0
for i in $(seq 1 180); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "pi05 SERVER DIED. Last lines of $SLOG:"; tail -n 40 "$SLOG"; exit 1; fi
  if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "pi05 server is up after $((i*5))s"; UP=1; break; fi
  sleep 5
done
[ "$UP" = 1 ] || { echo "pi05 server did not come up in 15 min. Last lines of $SLOG:"; tail -n 40 "$SLOG"; exit 1; }

$PY libero_safety_suites/run_demo_pi05.py --port "$PORT" --out "$VLA_HARM_ROOT/demo_pi05" "$@"
