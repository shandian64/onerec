#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher (torchrun) for Stage1 (freeze LLM, train itemic embeddings).
#
# Usage:
#   bash OpenOneRec/pretrain/examples/pretrain_stg1.sh
# Env overrides:
#   MODEL_DIR=/path/to/Qwen3-*_itemic
#   OUTPUT_DIR=/path/to/output_dir
#   DATASET_CONFIG=examples/dataset_config/pretrain.json
#   NPROC_PER_NODE=1
#
# 中文说明：
# - 这个脚本是“单机版启动命令”，用 torchrun 代替原来复杂的 mpirun/hostfile。
# - 默认单卡训练；如果你有多张 GPU，把 NPROC_PER_NODE 设成 GPU 数即可（例如 2/4/8）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${PRETRAIN_DIR}/.." && pwd)"

cd "${PRETRAIN_DIR}"

# 训练相关路径：都支持用环境变量覆盖，方便你在不同机器/目录复用
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/Qwen3-0.6B_itemic}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/models/output}"
DATASET_CONFIG="${DATASET_CONFIG:-examples/dataset_config/pretrain.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "${OUTPUT_DIR}" /tmp/_wids_cache

export PYTHONPATH="${PRETRAIN_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# dataset_config 默认要求 sources 指向一个“json 文件列表”（而不是直接写死 parquet 目录）。
# 这里如果检测到 file_list.json 不存在，就从 OpenOneRec/output/pretrain_*.parquet 自动生成一份。
# Keep dataset_config stable (expects a JSON file list). Generate it if missing.
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

# Stage1：冻结 LLM 参数，只训练 itemic token 的 embedding（start_optimize_embedding_index=151669）
torchrun --nproc_per_node="${NPROC_PER_NODE}" recipes/train_qwen3.py \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_config "${DATASET_CONFIG}" \
  --freeze_llm \
  --use_tie_weights \
  --start_optimize_embedding_index 151669
