# FU-TCM workflow overview

The current branch contains workflow code and configuration only. Source material, generated datasets, training splits, images, and evaluation samples must remain local in the ignored directories documented in the repository `.gitignore`.

## Classical text QA

- Code: `classical_text_qa/scripts/`
- Local inputs and outputs: `classical_text_qa/qa_output*/`, `classical_text_qa/sft_data/`
- Main flow: generate categorized QA pairs, clean them, then convert them to SFT format.

```bash
python classical_text_qa/scripts/generate/generate_qa_01.py
python classical_text_qa/scripts/process/clean_and_export.py
python classical_text_qa/scripts/process/convert_to_sft_format.py
```

## Textbook QA and illustrated-book VQA

- Code: `textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/`
- Local source books: `textbook_qa_and_book_vqa/tcm_vision_dataflow/source_books/`
- Local intermediates and outputs: `.../data/`, `.../results/`, plus the module-level `sft_data/` and `sft_image_data/`
- Main flow: extract text QA from PDFs and generate grounded VQA from illustrated TCM books.

## Clinical case reasoning

- Code: `clinical_case_reasoning/meta_reasoning/scripts/` and `clinical_case_reasoning/mlzy_reasoning/scripts/`
- Configuration: each subproject's `configs/` directory
- Local inputs and outputs: each subproject's ignored `data/` directory
- Main flow: extract cases, convert them into structured syndrome-differentiation reasoning, verify the output, and prepare local train/test splits.

## SFT, GRPO, and evaluation

- Code: `sft_grpo_training_evaluation/`
- Training configuration: `sft_grpo_training_evaluation/llamafactory_qwen35/`
- Benchmark builders: `sft_grpo_training_evaluation/benchmark_tools/`
- Local data: `sft_data/`, `sft_image_data/`, `sft_merged/`, `grpodata/`, `verl_data/`, and `benchmark/`
- Main flow: merge SFT inputs, convert reasoning data for verl, run GRPO, and evaluate a local model.

The ignored data directories are intentionally absent from Git and are created by the scripts or by the operator when needed.
