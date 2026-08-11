# FU-TCM training overview

FU-TCM uses supervised domain adaptation followed by reasoning alignment. The training target is the model's TCM knowledge, multimodal understanding, and syndrome-differentiation ability.

## Stage 1: Full-parameter SFT

The current Qwen3.5-9B configuration uses:

- LLaMA-Factory full-parameter fine-tuning;
- text and image instruction samples;
- DeepSpeed ZeRO-3 and bf16;
- 4,096-token context length;
- gradient checkpointing;
- a cosine learning-rate schedule.

```bash
cd sft_grpo_training_evaluation
bash llamafactory_qwen35/scripts/setup_llamafactory_env.sh
conda activate qwen35_ft
bash llamafactory_qwen35/scripts/train_full_ds3.sh
```

Configuration: `llamafactory_qwen35/qwen35_9b_full_sft_ds3.yaml`

## Stage 2: GRPO reasoning alignment

The GRPO stage aligns the model with a structured TCM reasoning target. Its reward evaluates:

1. required reasoning sections and output format;
2. agreement between predicted and reference syndromes;
3. consistency of eight-principle, organ, qi-blood-fluid, and pathogenic-factor judgments.

```bash
cd sft_grpo_training_evaluation
bash run_tcm_grpo_smoke.sh
```

The main reward entry point is `reward_functions.py::compute_score`.

## Stage 3: Evaluation

FU-TCM includes equivalent evaluation paths for Transformers and vLLM:

```bash
cd sft_grpo_training_evaluation
TCM_MODEL_PATH=/path/to/FU-TCM python eval_qwen35.py
TCM_MODEL_PATH=/path/to/FU-TCM python eval_qwen35_vllm.py
```

Evaluation reports overall accuracy and grouped accuracy by dataset and category. Benchmark tools can also generate blinded material for clinician review.
