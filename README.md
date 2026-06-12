# TCM_final

这是 Fu-TCM 数据生产、训练和评测目录。现在按四段流程组织：古籍 QA、教材 PDF/带图书籍 VQA、两类病例推理数据扩写，然后进入 SFT 与 GRPO。

浏览版入口：`index.html`；架构总览：`docs/architecture_overview.html`；更详细的四段流程索引：`docs/workflow_overview.html`。

## 一、古籍 QA 对提取处理

- 生成脚本：`scripts/generate/generate_qa_01.py` 到 `generate_qa_07.py`
- 清洗转换：`scripts/process/clean_and_export.py`、`scripts/process/convert_to_sft_format.py`
- 主要产物：`qa_output/`、`sft_data/guji700.json`、`sft_data/guji700_sampled_1000.jsonl`

## 二、教材 PDF QA 与带图书籍 VQA 提取处理

- 教材 PDF：`tcm_vision_dataflow/source_books/`、`tcm_vision_dataflow/workflows/dataflow2/scripts/qa/medical_pdf_to_qa_pipeline.py`
- 带图 VQA：`tcm_vision_dataflow/workflows/dataflow2/scripts/vqa/`、`tcm_vision_dataflow/data/`、`tcm_vision_dataflow/results/`
- 主要产物：`tcm_vision_dataflow/data/medical_books_qa/`、`sft_data/jiaocai.jsonl`、`sft_image_data/`

## 三、推理数据生成：两类病例数据扩写

- 名老中医病例：`mlzy_reasoning/`，从 CSV/XLSX 抽取、模型扩写、复核、切分
- meta reasoning 病案：`meta_reasoning/`，从 JSON 转为结构化辨证推理数据
- 主要产物：`mlzy_reasoning/data/processed/`、`meta_reasoning/data/processed/`、`grpodata/bianzheng_*_train.jsonl`

## 四、SFT、GRPO、评测与训练

- SFT：`sft_data/`、`sft_image_data/`、`sft_merged/`、`prepare_tcm_multimodal_data.py`
- 训练：`llamafactory_qwen35/`
- GRPO：`grpodata/`、`convert_bianzheng_to_verl_grpo.py`、`reward_functions.py`、`run_tcm_grpo_smoke*.sh`
- 评测：`benchmark/`、`eval_qwen35.py`、`eval_qwen35_vllm.py`

## 环境变量

```bash
export TCM_API_KEY="<your_api_key>"
export DF_API_KEY="<your_api_key>"
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="https://cc.580ai.net/v1"
export OPENAI_MODEL="claude-opus-4-6"
```

## GitHub 上传提醒

普通 GitHub 仓库单文件超过 100 MiB 会被拒绝。上传前请看 `docs/large_files.tsv`；大文件建议用 Git LFS、Release assets，或只上传脚本和小样本。
