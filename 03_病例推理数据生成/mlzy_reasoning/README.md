# 名老中医辨证推理数据流程

本目录整理了“名老中医”数据从原始 CSV 到辨证推理训练数据的处理流程。脚本已改为基于本目录自动定位路径，不再依赖下载目录或脚本所在目录。

## 目录结构

```text
名老中医/                         原始名老中医数据目录
configs/mapping_table.json         辨证映射规则
prompt_example.json                转换示例
scripts/extract/                   从 CSV 提取 raw_data
scripts/convert/                   调用模型转换结构化辨证数据
scripts/verify/                    四诊补足与辨证参数复核
scripts/prepare/                   转为 verl/parquet 数据
data/processed/                    中间与最终 JSON/JSONL 数据
data/train/train_grpo_600.json     prepare 阶段训练输入
data_parquet/                      prepare 脚本默认输出目录
```

## 环境变量

脚本不保存明文 API key。运行需要调用模型的脚本前，请设置环境变量：

```bash
export OPENAI_API_KEY="你的API_KEY"
export OPENAI_BASE_URL="https://api.minimax.chat/v1"
export OPENAI_MODEL="MiniMax-M2.7"
```

`scripts/verify/verify_mlzy.py` 也支持 Claude/Anthropic 兼容环境变量：

```bash
export ANTHROPIC_API_KEY="你的API_KEY"
export ANTHROPIC_BASE_URL="https://cc.580ai.net"
export ANTHROPIC_MODEL="claude-opus-4-6"
```

## 运行顺序

```bash
python scripts/extract/extract_minglaoyishi.py
python scripts/convert/convert_minimax.py
python scripts/verify/verify_mlzy.py
python scripts/prepare/prepare_data.py
```

## 主要数据

```text
data/processed/名老中医_extracted.json
data/processed/bianzheng_mlzy.jsonl
data/processed/bianzheng_mlzy_train.jsonl
data/processed/bianzheng_mlzy_test.jsonl
```
