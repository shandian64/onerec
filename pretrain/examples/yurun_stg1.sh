#!/usr/bin/env bash
set -euo pipefail

# Backwards-compatible alias for the simplified Stage1 launcher.
# 中文说明：保留老文件名，避免你之前的笔记/脚本引用失效；实际会调用 pretrain_stg1.sh。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/pretrain_stg1.sh"
