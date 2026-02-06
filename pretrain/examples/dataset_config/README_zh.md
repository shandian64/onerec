# dataset_config 中文说明

这两个 JSON 文件会被 `OpenOneRec/pretrain/recipes/train_qwen3.py` 读取，并传给 DataLoader。

## 文件

- `pretrain.json`：预训练（Stage1/Stage2）使用的配置（通常对应 `OpenOneRec/output/pretrain_*.parquet`）。
- `sft.json`：SFT（post-training）使用的配置（通常对应 `OpenOneRec/output/sft_*.parquet`）。

## 关键字段（高频会改）

- `sources`：一个 JSON 文件路径，里面是 parquet 文件路径列表（不是目录）。本仓库里默认是：
  - `OpenOneRec/output/split_data_pretrain/file_list.json`
  - `OpenOneRec/output/split_data_sft/file_list.json`
- `base_model_dir`：tokenizer/config 的来源模型目录（建议用相对路径，便于搬机器）。
- `max_length` / `max_sample_length`：样本截断/拼接相关长度。
- `num_workers`：DataLoader worker 数。
- `num_epochs`：对文件列表重复的 epoch 次数（会影响训练看到的数据量）。
- `add_think_pattern`（仅 `sft.json` 默认开启）：会自动给 user prompt 加 `/think` 或 `/no_think`，并补齐 `<think></think>`。

## 和启动脚本的关系

`OpenOneRec/pretrain/examples/*.sh` 会在缺少 `file_list.json` 时自动从 `OpenOneRec/output/*.parquet` 生成一份列表，避免你手动写。
