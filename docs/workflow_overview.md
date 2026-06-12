# TCM_final 四段流程总览

## 1. 古籍 QA 对提取处理

从中医古籍分类文本生成 QA，对结果清洗、去重、合并，并转换为 SFT 可用格式。

### 输入与来源

- `scripts/generate/`：7 类古籍 QA 生成脚本
- `qa_output/`：按 herbal/formulas/classics/clinical/diagnostics/wellness/theory 分类的生成结果

### 脚本与处理

- `scripts/generate/generate_qa_01.py`：中药类 QA 生成
- `scripts/generate/generate_qa_02.py`：方剂类 QA 生成
- `scripts/generate/generate_qa_03.py`：经典类 QA 生成
- `scripts/generate/generate_qa_04.py`：临床类 QA 生成
- `scripts/generate/generate_qa_05.py`：诊断类 QA 生成
- `scripts/generate/generate_qa_06.py`：养生类 QA 生成
- `scripts/generate/generate_qa_07.py`：理论类 QA 生成
- `scripts/process/clean_and_export.py`：清洗并导出 QA
- `scripts/process/convert_to_sft_format.py`：转换为 SFT 数据格式

### 产物

- `sft_data/guji700.json`：古籍 QA SFT 主数据
- `sft_data/guji700_remaining.json`：剩余古籍数据
- `sft_data/guji700_sampled_1000.jsonl`：抽样评测/转换用 JSONL

### 常用命令

```bash
python scripts/generate/generate_qa_01.py
python scripts/process/clean_and_export.py
python scripts/process/convert_to_sft_format.py
```

## 2. 教材 PDF QA 与带图书籍 VQA 提取处理

把教材 PDF 书籍抽成文本 QA，同时把带图书籍、舌诊、望诊、脉诊、中药草图等图文材料抽成 VQA。

### 输入与来源

- `tcm_vision_dataflow/source_books/`：原始 PDF/EPUB 书籍
- `tcm_vision_dataflow/data/`：医学书籍 QA、舌诊/望诊/脉诊/中药草图 VQA 数据
- `tcm_vision_dataflow/results/`：DataFlow 已生成的图文 VQA 结果

### 脚本与处理

- `tcm_vision_dataflow/workflows/dataflow2/scripts/qa/medical_pdf_to_qa_pipeline.py`：教材 PDF 到医学 QA 流水线
- `tcm_vision_dataflow/workflows/dataflow2/scripts/qa/gen_ypd2_qa.py`：药典二部 QA 生成
- `tcm_vision_dataflow/workflows/dataflow2/scripts/qa/gen_ypd34_qa.py`：药典三/四部 QA 生成
- `tcm_vision_dataflow/workflows/dataflow2/scripts/vqa/`：带图书籍与图像 VQA 生成脚本集合
- `tcm_vision_dataflow/workflows/dataflow_runtime/`：DataFlow 框架源码

### 产物

- `tcm_vision_dataflow/data/medical_books_qa/`：教材/医学书籍 QA 输出
- `tcm_vision_dataflow/data/shezhen_vqa/`：舌诊 VQA 数据
- `tcm_vision_dataflow/data/maizhen_vqa/`：脉诊/望诊相关 VQA 数据
- `tcm_vision_dataflow/data/zhongyao_caotu_vqa/`：中药草图 VQA 数据
- `sft_data/jiaocai.jsonl`：教材文本 QA 转成的 SFT 数据
- `sft_image_data/`：图文/望诊/舌诊等 SFT 图像数据

### 常用命令

```bash
cd tcm_vision_dataflow/workflows/dataflow2/scripts/qa
python medical_pdf_to_qa_pipeline.py
python ../vqa/gen_shezhen_vqa_final.py
```

## 3. 推理数据生成，两类病例数据扩写

把两类病例数据扩写/转换为结构化辨证推理数据，并进行四诊补足、证候参数复核和 train/test 切分。

### 输入与来源

- `mlzy_reasoning/名老中医/`：名老中医原始 CSV/XLSX 病例数据
- `meta_reasoning/data/raw/meta_reasoning.json`：meta reasoning 原始病案数据
- `mlzy_reasoning/configs/mapping_table.json`：名老中医辨证映射规则
- `meta_reasoning/configs/mapping_table.json`：meta reasoning 辨证映射规则

### 脚本与处理

- `mlzy_reasoning/scripts/extract/extract_minglaoyishi.py`：名老中医 CSV 抽取
- `mlzy_reasoning/scripts/convert/convert_minimax.py`：名老中医病例扩写/结构化转换
- `mlzy_reasoning/scripts/verify/verify_mlzy.py`：名老中医四诊与辨证复核
- `mlzy_reasoning/scripts/prepare/prepare_data.py`：名老中医 GRPO/SFT 数据准备
- `meta_reasoning/scripts/convert/convert_bianzheng.py`：meta reasoning 病案扩写/结构化转换
- `meta_reasoning/scripts/verify/verify_bianzheng.py`：meta reasoning 辨证复核

### 产物

- `mlzy_reasoning/data/processed/bianzheng_mlzy.jsonl`：名老中医结构化辨证数据
- `mlzy_reasoning/data/processed/bianzheng_mlzy_train.jsonl`：名老中医训练集
- `mlzy_reasoning/data/processed/bianzheng_mlzy_test.jsonl`：名老中医测试集
- `meta_reasoning/data/processed/bianzheng_minimax.jsonl`：meta reasoning 结构化辨证训练数据
- `meta_reasoning/data/processed/bianzheng_minimax_test.jsonl`：meta reasoning 测试集
- `grpodata/bianzheng_mlzy_train.jsonl`：GRPO 名老中医训练数据
- `grpodata/bianzheng_xin_train.jsonl`：GRPO 另一类病案训练数据
- `grpodata/bianzheng_merged_train.jsonl`：合并后的 GRPO 训练数据

### 常用命令

```bash
cd mlzy_reasoning && python scripts/extract/extract_minglaoyishi.py
cd mlzy_reasoning && python scripts/convert/convert_minimax.py
cd meta_reasoning && python scripts/convert/convert_bianzheng.py
```

## 4. SFT、GRPO、评测与训练入口

把前面三部分产物汇入 SFT、多模态 SFT、冷启动和 GRPO 数据，并连接 LLaMA-Factory、verl、reward 和评测脚本。

### 输入与来源

- `sft_data/`：文本 SFT 数据：古籍、教材、辨证、身份数据等
- `sft_image_data/`：图像/图文 SFT 数据
- `sft_merged/`：合并后的多模态 SFT 数据
- `grpodata/`：辨证 GRPO、coldstart、train/test 数据
- `benchmark/`：评测集、图像评测和医生盲测材料

### 脚本与处理

- `prepare_tcm_multimodal_data.py`：准备/合并多模态训练数据
- `merge_identity_overwrite_sft.py`：合并身份认知 SFT 数据
- `llamafactory_qwen35/`：Qwen3.5 / LLaMA-Factory 训练配置
- `convert_bianzheng_to_verl_grpo.py`：辨证数据转 verl GRPO 格式
- `reward_functions.py`：GRPO reward 函数
- `generate_coldstart_400.py`：生成 coldstart 采样数据
- `remove_coldstart_from_train.py`：从训练集剔除 coldstart 样本
- `run_tcm_grpo_smoke.sh`：GRPO smoke 测试
- `run_tcm_grpo_smoke_hf.sh`：HuggingFace 版本 GRPO smoke 测试
- `eval_qwen35.py`：本地模型评测
- `eval_qwen35_vllm.py`：vLLM 评测

### 产物

- `llamafactory_qwen35/qwen35_9b_full_sft_ds3.yaml`：SFT 训练配置
- `grpodata/verl/bianzheng_grpo_train.jsonl`：verl GRPO 训练数据
- `grpodata/verl/bianzheng_grpo_test.jsonl`：verl GRPO 测试数据
- `benchmark/`：评测输入与结果承载目录

### 常用命令

```bash
python prepare_tcm_multimodal_data.py
python convert_bianzheng_to_verl_grpo.py
bash run_tcm_grpo_smoke.sh
python eval_qwen35.py
```
