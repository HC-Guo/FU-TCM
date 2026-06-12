# Meta Reasoning 辨证数据流程

本目录整理了 `meta_reasoning.json` 到结构化辨证数据的转换、测试集结果和辨证参数复核流程。脚本已改为基于本目录自动定位路径，不再依赖下载目录或运行时所在目录。

## 目录结构

```text
data/raw/meta_reasoning.json              原始 meta reasoning 病案数据
data/processed/bianzheng_minimax.jsonl    已转换训练集结果
data/processed/bianzheng_minimax_test.jsonl 已转换测试集结果
configs/mapping_table.json                辨证映射规则
scripts/convert/convert_bianzheng.py      多模型结构化辨证转换脚本
scripts/verify/verify_bianzheng.py        辨证参数复核脚本
```

## 环境变量

脚本不保存明文 API key。运行转换脚本前设置：

```bash
export OPENAI_API_KEY="你的API_KEY"
export OPENAI_BASE_URL="https://cc.580ai.net/v1"
export META_REASONING_MODELS="claude-sonnet-4-6,deepseek-reasoner"
```

运行复核脚本可使用：

```bash
export ANTHROPIC_API_KEY="你的API_KEY"
export ANTHROPIC_BASE_URL="https://cc.580ai.net"
export ANTHROPIC_MODEL="claude-opus-4-6"
```

## 运行顺序

```bash
python scripts/convert/convert_bianzheng.py
python scripts/verify/verify_bianzheng.py
```

转换脚本读取 `data/raw/meta_reasoning.json`，输出到 `data/processed/`。复核脚本读取 `data/processed/bianzheng_minimax.jsonl` 和 `data/processed/bianzheng_minimax_test.jsonl`，输出对应的 `*_verified.jsonl` 文件。
