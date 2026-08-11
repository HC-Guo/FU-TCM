# 中医视觉/图文 DataFlow 数据处理流程

本目录保留中医视觉/图文数据处理代码、提示词与项目定制的 DataFlow 1.0.10 runtime。原始材料和生成数据不进入版本控制。

## 目录结构

```text
source_books/                 舌诊、望诊、面诊等原始 PDF、EPUB 书籍
archives/                     原始压缩包归档
data/                         医学书籍 QA、舌诊/望诊/脉诊/中药草图 VQA 数据
results/                      DataFlow 已生成的中医视觉/图文 VQA 结果
workflows/dataflow2/          FU-TCM 的脚本与 prompt
workflows/dataflow_runtime/   项目定制的 DataFlow 1.0.10 代码（不含 example/test）
requirements.txt             本地 DataFlow runtime 及通用 Python 依赖
```

## 数据说明

- `source_books/`、`archives/`、`data/` 和 `results/` 均为本地目录，并由仓库根目录的 `.gitignore` 排除。
- 当前分支不包含书籍、图像、QA/VQA 记录、压缩包或生成结果。
- 目录名使用 `tcm_vision_dataflow`，覆盖舌诊、望诊、面诊、脉诊、中药草图和医学书籍 QA 等工作流。

## API 环境变量

脚本中未保存真实 API key。若运行 DataFlow 或 VQA 生成脚本，可按脚本要求设置：

```bash
export DF_API_KEY="<your_api_key>"
export MINIMAX_API_URL="https://api.minimax.chat/v1/chat/completions"
export MINIMAX_MODEL="MiniMax-M1"
```

```bash
cd textbook_qa_and_book_vqa/tcm_vision_dataflow
pip install -r requirements.txt
```

## 主要入口

```text
workflows/dataflow2/scripts/vqa/
workflows/dataflow2/scripts/qa/
workflows/dataflow2/prompt_engineering/
workflows/dataflow_runtime/
```

## 提交提醒

提交前运行 `git status --short`，确认本地图片、PDF、JSONL 和压缩包没有进入暂存区。若需要发布数据，请使用独立的数据仓库或受控存储。
