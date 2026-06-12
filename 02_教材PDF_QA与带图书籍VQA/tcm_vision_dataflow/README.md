# 中医视觉/图文 DataFlow 数据处理流程

本目录合并整理自微信接收的 `Dataflow 2/`、`DataFlow/`、`data/`、`舌诊带图书籍/` 和 `舌诊带图书籍.zip`。

## 目录结构

```text
source_books/                 舌诊、望诊、面诊等原始 PDF、EPUB 书籍
archives/                     原始压缩包归档
data/                         医学书籍 QA、舌诊/望诊/脉诊/中药草图 VQA 数据
results/                      DataFlow 已生成的中医视觉/图文 VQA 结果
workflows/dataflow2/          Dataflow 2 中的脚本、prompt 和框架
workflows/dataflow_runtime/   DataFlow 框架源码
manifest/                     来源清单和整理记录
```

## 合并说明

- 外层 `data/` 与 `Dataflow 2/data/` 完全一致，整理后只保留一份在 `data/`。
- `DataFlow/DataFlow/.git`、`__pycache__`、`.DS_Store`、`.claude` 等本地或缓存文件未纳入整理目录。
- 原始 zip 保留在 `archives/`，已解压书籍保留在 `source_books/`。
- 目录名使用 `tcm_vision_dataflow`，覆盖舌诊、望诊、面诊、脉诊、中药草图和医学书籍 QA 等内容。

## API 环境变量

脚本中未保存真实 API key。若运行 DataFlow 或 VQA 生成脚本，可按脚本要求设置：

```bash
export DF_API_KEY="<your_api_key>"
export MINIMAX_API_URL="https://api.minimax.chat/v1/chat/completions"
export MINIMAX_MODEL="MiniMax-M1"
```

## 主要入口

```text
workflows/dataflow2/scripts/vqa/
workflows/dataflow2/scripts/qa/
workflows/dataflow2/prompt_engineering/
workflows/dataflow_runtime/
```

## 上传提醒

本目录包含大量图片、PDF、JSONL 和 zip 文件。普通 GitHub 仓库单文件超过 100 MiB 会被拒绝；上传前请查看 `../docs/large_files.tsv`，大文件建议使用 Git LFS 或 Release assets。
