# FU-TCM architecture overview

FU-TCM connects domain-source processing, structured reasoning, model training, and evaluation through four modular workflows.

```text
TCM source materials
  ├─ classical texts and generated QA checkpoints
  ├─ textbooks, illustrated books, and extracted images
  └─ clinical cases
                 │
                 ▼
Processing workflows
  ├─ classical_text_qa
  ├─ textbook_qa_and_book_vqa
  └─ clinical_case_reasoning
                 │
                 ▼
Model-ready datasets
  ├─ text and multimodal SFT datasets
  ├─ structured reasoning and GRPO datasets
  └─ evaluation samples and generated reports
                 │
                 ▼
Training and evaluation workflows
  └─ sft_grpo_training_evaluation
```

## Design principles

1. Each workflow can be installed and run independently.
2. Text, multimodal, and reasoning pipelines share clear intermediate formats.
3. Training and evaluation configuration remains reproducible across environments.
4. Service credentials are supplied through environment variables.

See `workflow_overview.md` for executable entry points and expected inputs and outputs.
