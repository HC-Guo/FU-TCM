# TCM_final 架构总览

TCM_final 可以理解为一个面向 Fu-TCM 的数据工程与训练工程仓库：前半部分负责把古籍、教材、图文书籍和病例数据加工成训练/评测数据，后半部分负责 SFT、GRPO、评测与训练配置。

## 分层架构

```text
原始数据层
  ├─ 古籍文本与分类输出：qa_output/
  ├─ 教材 PDF、带图书籍、图像材料：tcm_vision_dataflow/source_books/、data/
  ├─ 名老中医 CSV/XLSX：mlzy_reasoning/名老中医/
  └─ meta reasoning 病案：meta_reasoning/data/raw/meta_reasoning.json

抽取与扩写层
  ├─ 古籍 QA 生成：scripts/generate/
  ├─ 古籍 QA 清洗/SFT 转换：scripts/process/
  ├─ 教材 PDF QA：tcm_vision_dataflow/workflows/dataflow2/scripts/qa/
  ├─ 图文 VQA：tcm_vision_dataflow/workflows/dataflow2/scripts/vqa/
  ├─ 名老中医病例扩写：mlzy_reasoning/scripts/
  └─ meta reasoning 病例扩写：meta_reasoning/scripts/

数据产物层
  ├─ 文本 SFT：sft_data/
  ├─ 图文 SFT：sft_image_data/
  ├─ 多模态合并：sft_merged/
  ├─ 推理/GRPO 数据：grpodata/
  └─ 评测数据：benchmark/

训练与评测层
  ├─ SFT 配置：llamafactory_qwen35/
  ├─ GRPO 转换：convert_bianzheng_to_verl_grpo.py
  ├─ Reward：reward_functions.py
  ├─ GRPO smoke：run_tcm_grpo_smoke*.sh
  └─ 模型评测：eval_qwen35.py、eval_qwen35_vllm.py
```

## 四条主数据流

1. 古籍 QA 流：`scripts/generate/` -> `qa_output/` -> `scripts/process/` -> `sft_data/guji700*.json*`
2. 教材/图文流：`tcm_vision_dataflow/source_books,data,results` -> DataFlow QA/VQA 脚本 -> `sft_data/jiaocai.jsonl`、`sft_image_data/`
3. 病例推理流：`mlzy_reasoning/` + `meta_reasoning/` -> 结构化辨证扩写/复核 -> `grpodata/bianzheng_*`
4. 训练评测流：`sft_data/` + `sft_image_data/` + `grpodata/` -> `llamafactory_qwen35/`、verl GRPO、`benchmark/`

## 核心入口

- 流程总览：`docs/workflow_overview.html`
- 四段流程清单：`docs/workflow_manifest.tsv`
- 古籍 QA：`docs/01_guji_qa.html`
- 教材 PDF / 图文 VQA：`docs/02_pdf_vqa.html`
- 病例推理扩写：`docs/03_reasoning_expansion.html`
- SFT / GRPO：`docs/04_sft_grpo.html`
