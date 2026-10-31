#!/usr/bin/env bash
set -e
export HF_ENDPOINT=https://hf-mirror.com

echo "[cube] Downloading dataset..."
/data/chenyizi/le-wm/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('quentinll/lewm-cube', repo_type='dataset', local_dir='/data/chenyizi/le-wm/datasets/lewm-cube')
print('dataset done')
"

echo "[cube] Downloading checkpoint..."
/data/chenyizi/le-wm/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('quentinll/lewm-cube', repo_type='model', local_dir='/data/chenyizi/le-wm/checkpoints/lewm-cube')
print('checkpoint done')
"

echo "[cube] All done."
