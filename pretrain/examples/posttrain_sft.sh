#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher (torchrun) for SFT (post-training).
#
# Usage:
#   bash OpenOneRec/pretrain/examples/posttrain_sft.sh
# Env overrides:
#   MODEL_DIR=/path/to/stage2_converted_or_base_model
#   OUTPUT_DIR=/path/to/output_dir
#   DATASET_CONFIG=examples/dataset_config/sft.json
#   NPROC_PER_NODE=1
#
# 中文说明：
# - 这里的 SFT 指 post-training 的监督微调阶段（一般用 sft_*.parquet）。
# - 同样用 torchrun 单机启动，避免原来 mpirun/hostfile 的复杂环境依赖。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${PRETRAIN_DIR}/.." && pwd)"

cd "${PRETRAIN_DIR}"

# 训练相关路径：都支持用环境变量覆盖
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/Qwen3-0.6B_itemic}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/models/output}"
DATASET_CONFIG="${DATASET_CONFIG:-examples/dataset_config/sft.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "${OUTPUT_DIR}" /tmp/_wids_cache

export PYTHONPATH="${PRETRAIN_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# dataset_config 默认要求 sources 指向一个“json 文件列表”，所以这里也做了自动生成。
FILE_LIST_PATH="${PROJECT_DIR}/output/split_data_sft/file_list.json"
if [[ ! -f "${FILE_LIST_PATH}" ]]; then
  export PROJECT_DIR FILE_LIST_PATH
  python3 - <<'PY'
import glob
import json
import os
import sys

project_dir = os.environ["PROJECT_DIR"]
file_list_path = os.environ["FILE_LIST_PATH"]

parquets = sorted(glob.glob(os.path.join(project_dir, "output", "sft_*.parquet")))
if not parquets:
    sys.stderr.write(f"Error: no files matched {os.path.join(project_dir, 'output', 'sft_*.parquet')}\n")
    sys.exit(1)

pretrain_dir = os.getcwd()
parquets = [os.path.relpath(p, pretrain_dir) for p in parquets]

os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
with open(file_list_path, "w", encoding="utf-8") as f:
    json.dump(parquets, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(parquets)} parquet paths to {file_list_path}")
PY
fi

# 启动训练：核心参数都在 dataset_config 里（如 max_length/num_workers/num_epochs/add_think_pattern 等）
torchrun --nproc_per_node="${NPROC_PER_NODE}" recipes/train_qwen3.py \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_config "${DATASET_CONFIG}" \
  --use_tie_weights
