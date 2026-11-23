#!/usr/bin/env bash
# One-time setup on a fresh GPU box (tested on RunPod, runpod/pytorch ubuntu24.04 image, A40). Run from the repo root.
#   bash scripts/setup_pod.sh render     # clones + render venv (.venv-render, Python 3.10) + base LIBERO assets
#   bash scripts/setup_pod.sh openpi     # openpi checkout + its venv (local disk, $OPENPI_VENV, default /root/openpi-venv) with patched transformers
#   bash scripts/setup_pod.sh convert pi05_base     # download a JAX checkpoint and convert it to PyTorch under cache/ckpt/
# Needs HF_TOKEN in the environment (PaliGemma is gated) for extraction.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install -q uv
export UV_LINK_MODE=copy   # keep the transformers patch out of the shared uv cache
OPENPI_VENV=${OPENPI_VENV:-/root/openpi-venv}
PY=$OPENPI_VENV/bin/python

case "${1:-}" in
render)
  [ -d cache/LIBERO-Safety ] || git clone -q --depth 1 https://github.com/LIBERO-SAFETY/LIBERO-Safety.git cache/LIBERO-Safety
  [ -d cache/LIBERO ] || git clone -q --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git cache/LIBERO
  [ -d cache/libero_assets ] || cp -R cache/LIBERO/libero/libero/assets cache/libero_assets
  apt-get install -y -qq libegl1 libgl1 libosmesa6 >/dev/null 2>&1 || true
  uv venv -q --python 3.10 .venv-render
  VIRTUAL_ENV=.venv-render uv pip install -q "numpy<2" mujoco==3.2.3 -e cache/LIBERO-Safety/third_party/robosuite-1.4 bddl==1.0.1 \
      easydict pyyaml imageio opencv-python-headless matplotlib gym==0.25.2 cloudpickle future h5py \
      termcolor numba scipy pillow pynput scikit-image remotezip
  ;;
openpi)
  [ -d cache/openpi ] || git clone -q https://github.com/Physical-Intelligence/openpi.git cache/openpi
  # RunPod's /workspace is a network filesystem: importing a venv from it takes minutes, so put the venv on local disk
  export UV_PROJECT_ENVIRONMENT=$OPENPI_VENV
  (cd cache/openpi && git checkout -q 215abfb && git submodule update -q --init --recursive \
     && GIT_LFS_SKIP_SMUDGE=1 uv sync -q --no-install-package rerun-sdk)
  SP=$(ls -d "$OPENPI_VENV"/lib/python3.*/site-packages)
  cp -r cache/openpi/src/openpi/models_pytorch/transformers_replace/* "$SP/transformers/"
  "$OPENPI_VENV"/bin/python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print('openpi env OK')"
  ;;
convert)
  name=${2:?checkpoint name, e.g. pi05_base}
  out=cache/ckpt/${name}_pytorch
  [ -f "$out/model.safetensors" ] && { echo "already converted: $out"; exit 0; }
  export OPENPI_DATA_HOME=$PWD/cache/openpi_data
  case "$name" in
    pi05_libero_safety) src=$PWD/cache/ckpt/pi05_libero_safety_jax
       "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('LIBERO-Safety/pi05_libero_safety', local_dir='$src')" ;;
    *) src=$("$PY" -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/$name'))") ;;
  esac
  (cd cache/openpi && "$PY" examples/convert_jax_model_to_pytorch.py --checkpoint_dir "$src" \
     --config_name pi05_libero --output_path "$OLDPWD/$out" --precision bfloat16)
  ;;
*) echo "usage: $0 render|openpi|convert <name>"; exit 1 ;;
esac
