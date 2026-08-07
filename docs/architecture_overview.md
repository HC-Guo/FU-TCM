# FU-TCM architecture overview

FU-TCM separates versioned workflow code from local-only research data.

```text
Local source data (ignored)
  ├─ classical texts and generated QA checkpoints
  ├─ textbooks, illustrated books, and extracted images
  └─ clinical cases
                 │
                 ▼
Versioned processing code
  ├─ 01_classical_text_qa
  ├─ 02_textbook_qa_and_book_vqa
  └─ 03_clinical_case_reasoning
                 │
                 ▼
Local model-ready data (ignored)
  ├─ text and multimodal SFT datasets
  ├─ structured reasoning and GRPO datasets
  └─ evaluation samples and generated reports
                 │
                 ▼
Versioned training and evaluation code
  └─ 04_sft_grpo_training_evaluation
```

## Design rules

1. Repository paths use English names.
2. Data-producing scripts and model configuration are versioned.
3. Source data, generated records, images, archives, train/test splits, and benchmarks stay in ignored local directories.
4. Secrets are supplied through environment variables and never committed.

See `workflow_overview.md` for executable entry points and `.gitignore` for the authoritative list of local-only paths.
