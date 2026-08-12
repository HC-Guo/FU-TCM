# FU-TCM training overview

FU-TCM uses a two-stage strategy: TCM domain-specific learning followed by Bianzheng-Grounded Policy Optimization.

## Stage 1: TCM domain-specific learning

Qwen3.5-9B and Qwen3.6-27B undergo full-parameter supervised fine-tuning on three complementary inputs:

- 1.53 million text question-answer examples;
- 8,331 image-text visual question-answer examples;
- 1,867 structured case-reasoning examples.

This stage establishes TCM knowledge, multimodal understanding, and structured *bianzheng* outputs. The executable 9B configuration in this repository uses LLaMA-Factory, DeepSpeed ZeRO-3, bf16, a 4,096-token context length, gradient checkpointing, and a cosine learning-rate schedule.

```bash
cd sft_grpo_training_evaluation
bash llamafactory_qwen35/scripts/setup_llamafactory_env.sh
conda activate qwen35_ft
bash llamafactory_qwen35/scripts/train_full_ds3.sh
```

Configuration: `llamafactory_qwen35/qwen35_9b_full_sft_ds3.yaml`

## Stage 2: BGPO reasoning alignment

Bianzheng-Grounded Policy Optimization evaluates:

1. the required response sections, order, and format;
2. tree-based similarity between predicted and reference syndromes;
3. fidelity across 30 intermediate fields covering the eight principles, organ systems, qi-blood-fluid states, and pathogenic factors.

The repository implements this objective with verl and the project-specific `reward_functions.py::compute_score` entry point.

```bash
cd sft_grpo_training_evaluation
bash run_tcm_grpo_smoke.sh
```

## Evaluation

FU-TCM provides Transformers and vLLM evaluation paths:

```bash
cd sft_grpo_training_evaluation
TCM_MODEL_PATH=/path/to/FU-TCM python eval_qwen35.py
TCM_MODEL_PATH=/path/to/FU-TCM python eval_qwen35_vllm.py
```

The paper evaluates TCM reasoning, text, and vision across six benchmarks and includes same-question external validation of model-only, independent physician, and FU-TCM-assisted responses.
