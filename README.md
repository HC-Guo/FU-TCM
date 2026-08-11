<h1 align="center">FU-TCM</h1>

<p align="center">
  <strong>Traditional Chinese Medicine Data &amp; Training Workflows</strong><br>
  面向中医古籍、教材图文、临床病例推理与模型训练的代码工作流
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white">
  <img alt="TCM AI workflows" src="https://img.shields.io/badge/domain-TCM%20AI-176B4D">
  <a href="SECURITY.md"><img alt="Security policy" src="https://img.shields.io/badge/security-policy-6B7280"></a>
</p>

<p align="center">
  <a href="index.html">项目导航</a> ·
  <a href="docs/architecture_overview.md">架构说明</a> ·
  <a href="docs/workflow_overview.md">工作流</a> ·
  <a href="SECURITY.md">安全政策</a>
</p>

## Overview

FU-TCM 将中医数据生产、结构化辨证推理、SFT/GRPO 训练准备和评测入口整理为四个相互衔接的模块，为中医人工智能研究提供可复用的端到端工作流。

### Key features

- **古籍文本 QA**：批量生成、清洗并转换为 SFT 格式。
- **教材与图文 VQA**：支持 PDF 文本 QA、舌诊/面诊/脉诊及中药图文问答。
- **临床辨证推理**：病例抽取、结构化转换、复核及确定性数据切分。
- **训练与评测**：提供 LLaMA-Factory SFT、verl GRPO、规则奖励与 benchmark 工具。

## Workflow

```mermaid
flowchart LR
    A["Classical texts"] --> B["Text QA / SFT"]
    C["Textbooks & illustrated books"] --> D["Grounded VQA / multimodal SFT"]
    E["Clinical cases"] --> F["Structured reasoning"]
    B --> G["SFT training"]
    D --> G
    F --> H["GRPO data & reward"]
    G --> I["Evaluation"]
    H --> I

    classDef source fill:#f6f1e8,stroke:#b88a44,color:#3f3424;
    classDef process fill:#eef7f2,stroke:#4f8f73,color:#173d2e;
    classDef train fill:#edf3ff,stroke:#6087c5,color:#1f365a;
    class A,C,E source;
    class B,D,F process;
    class G,H,I train;
```

## Repository layout

| Module | Purpose | Documentation |
| --- | --- | --- |
| `classical_text_qa/` | Classical TCM text QA generation, cleaning and SFT conversion | [Module guide](docs/classical_text_qa.html) |
| `textbook_qa_and_book_vqa/` | Textbook QA, illustrated-book VQA and the customized DataFlow runtime | [Module guide](docs/textbook_qa_and_book_vqa.html) |
| `clinical_case_reasoning/` | Structured syndrome-differentiation conversion and verification | [Module guide](docs/clinical_case_reasoning.html) |
| `sft_grpo_training_evaluation/` | SFT/GRPO preparation, reward functions, training and evaluation | [Module guide](docs/sft_grpo_evaluation.html) |

## Quick start

### Clone

```bash
git clone https://github.com/HC-Guo/FU-TCM.git
cd FU-TCM
```

### Install one workflow

只安装当前任务所需模块。教材/VQA 模块保留了项目定制的 DataFlow 1.0.10 runtime。

```bash
# Classical text QA
pip install -r classical_text_qa/requirements.txt

# Textbook QA and Book VQA
(
  cd textbook_qa_and_book_vqa/tcm_vision_dataflow
  pip install -r requirements.txt
)

# Clinical case reasoning
pip install -r clinical_case_reasoning/mlzy_reasoning/requirements.txt

# SFT, GRPO and evaluation
pip install -r sft_grpo_training_evaluation/requirements.txt
```

### Configure credentials

复制 `.env.example` 到本地环境文件，仅填写所选工作流需要的变量。不要提交真实密钥。

```bash
export TCM_API_KEY="<your_api_key>"
export DF_API_KEY="<your_api_key>"
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="<your_api_base_url>"
export OPENAI_MODEL="<your_model>"
```

### Run representative entry points

```bash
# Classical-text QA
python classical_text_qa/scripts/generate/generate_qa_01.py

# Clinical reasoning: extract -> convert -> verify -> split
python clinical_case_reasoning/mlzy_reasoning/scripts/extract/extract_minglaoyishi.py
python clinical_case_reasoning/mlzy_reasoning/scripts/convert/convert_minimax.py
python clinical_case_reasoning/mlzy_reasoning/scripts/verify/verify_mlzy.py
python clinical_case_reasoning/mlzy_reasoning/scripts/prepare/split_dataset.py

# Verified MLZY split -> verl GRPO
cd sft_grpo_training_evaluation
python convert_bianzheng_to_verl_grpo.py \
  --train ../clinical_case_reasoning/mlzy_reasoning/data/processed/bianzheng_mlzy_train.jsonl \
  --test ../clinical_case_reasoning/mlzy_reasoning/data/processed/bianzheng_mlzy_test.jsonl \
  --output-dir verl_data
```

## Documentation

| Guide | Contents |
| --- | --- |
| [Browsable project index](index.html) | Polished HTML navigation for the full repository |
| [Architecture overview](docs/architecture_overview.md) | Components and data flow across the four modules |
| [Workflow overview](docs/workflow_overview.md) | Main stages, entry points and expected inputs/outputs |
| [Security policy](SECURITY.md) | Credential and private-data handling rules |
