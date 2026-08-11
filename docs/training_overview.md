# FU-TCM training overview

FU-TCM uses a three-stage strategy that moves from broad TCM domain adaptation to structured case reasoning and reward-based alignment.

## Stage 1: Full-parameter domain SFT

Qwen3.5-9B and Qwen3.6-27B are adapted with text and visual question-answer data. The executable 9B configuration in this repository uses:

- LLaMA-Factory full-parameter fine-tuning;
- text and image instruction samples;
- DeepSpeed ZeRO-3 and bf16;
- a 4,096-token context length;
- gradient checkpointing;
- a cosine learning-rate schedule.

```bash
cd sft_grpo_training_evaluation
bash llamafactory_qwen35/scripts/setup_llamafactory_env.sh
conda activate qwen35_ft
bash llamafactory_qwen35/scripts/train_full_ds3.sh
```

Configuration: `llamafactory_qwen35/qwen35_9b_full_sft_ds3.yaml`

## Stage 2: Cold-start SFT

Structured clinical-case examples teach the model the required output format and the path from four-examination evidence through intermediate *bianzheng* fields to the final syndrome.

## Stage 3: BGPO reasoning alignment

Bianzheng-Grounded Policy Optimization evaluates:

1. the required response sections and output format;
2. consistency between the predicted and reference syndromes;
3. fidelity across the eight principles, organ systems, qi-blood-fluid states, pathogenic factors, and other intermediate fields.

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

Evaluation reports overall accuracy and grouped accuracy by dataset and category. Benchmark tools can also generate blinded materials for clinician review.

The paper evaluates case-based *bianzheng*, TCM text knowledge, examination questions, and visual understanding across three held-out internal benchmarks and three independently sourced tasks.
