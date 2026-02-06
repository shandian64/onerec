#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher (torchrun) for Stage2 (full-parameter co-pretraining).
#
# Usage:
#   bash OpenOneRec/pretrain/examples/pretrain_stg2.sh
# Env overrides:
#   MODEL_DIR=/path/to/stage1_converted_or_base_model
#   OUTPUT_DIR=/path/to/output_dir
#   DATASET_CONFIG=examples/dataset_config/pretrain.json
#   NPROC_PER_NODE=1
#
# 中文说明：
# - Stage2 是全参数训练（不再 --freeze_llm），通常用于混合“推荐数据 + 通用文本”的 co-pretrain。
# - 仍然是 torchrun 单机启动；多卡时设置 NPROC_PER_NODE。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${PRETRAIN_DIR}/.." && pwd)"

cd "${PRETRAIN_DIR}"

# 训练相关路径：都支持用环境变量覆盖
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/Qwen3-0.6B_itemic}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/models/output}"
DATASET_CONFIG="${DATASET_CONFIG:-examples/dataset_config/pretrain.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "${OUTPUT_DIR}" /tmp/_wids_cache

export PYTHONPATH="${PRETRAIN_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 同 Stage1：如果 file_list.json 不存在就自动生成（从 OpenOneRec/output/pretrain_*.parquet）。
FILE_LIST_PATH="${PROJECT_DIR}/output/split_data_pretrain/file_list.json"
if [[ ! -f "${FILE_LIST_PATH}" ]]; then
  export PROJECT_DIR FILE_LIST_PATH
  python3 - <<'PY'
import glob
import json
import os
import sys

project_dir = os.environ["PROJECT_DIR"]
file_list_path = os.environ["FILE_LIST_PATH"]

parquets = sorted(glob.glob(os.path.join(project_dir, "output", "pretrain_*.parquet")))
if not parquets:
    sys.stderr.write(f"Error: no files matched {os.path.join(project_dir, 'output', 'pretrain_*.parquet')}\n")
    sys.exit(1)

pretrain_dir = os.getcwd()
parquets = [os.path.relpath(p, pretrain_dir) for p in parquets]

os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
with open(file_list_path, "w", encoding="utf-8") as f:
    json.dump(parquets, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(parquets)} parquet paths to {file_list_path}")
PY
fi

# Stage2：全参数训练（注意：如果你是从 Stage1 的 converted 模型继续训练，把 MODEL_DIR 指到 converted 目录即可）
torchrun --nproc_per_node="${NPROC_PER_NODE}" recipes/train_qwen3.py \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_config "${DATASET_CONFIG}" \
  --use_tie_weights
