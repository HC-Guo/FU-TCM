# FU-TCM workflow overview

The project is organized as four connected workflows, from TCM source processing to supervised fine-tuning, reinforcement learning, and evaluation.

## Classical text QA

- Code: `classical_text_qa/scripts/`
- Inputs and outputs: `classical_text_qa/qa_output*/`, `classical_text_qa/sft_data/`
- Main flow: generate categorized QA pairs, clean them, then convert them to SFT format.

```bash
python classical_text_qa/scripts/generate/generate_qa_01.py
python classical_text_qa/scripts/process/clean_and_export.py
python classical_text_qa/scripts/process/convert_to_sft_format.py
```

## Textbook QA and illustrated-book VQA

- Code: `textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/`
- Source books: `textbook_qa_and_book_vqa/tcm_vision_dataflow/source_books/`
- Intermediates and outputs: `.../data/`, `.../results/`, plus the module-level `sft_data/` and `sft_image_data/`
- Main flow: extract text QA from PDFs and generate grounded VQA from illustrated TCM books.

## Clinical case reasoning

- Code: `clinical_case_reasoning/meta_reasoning/scripts/` and `clinical_case_reasoning/mlzy_reasoning/scripts/`
- Configuration: each subproject's `configs/` directory
- Inputs and outputs: each subproject's `data/` directory
- Main flow: extract cases, convert them into structured syndrome-differentiation reasoning, verify the output, and prepare train/test splits.

## SFT, GRPO, and evaluation

- Code: `sft_grpo_training_evaluation/`
- Training configuration: `sft_grpo_training_evaluation/llamafactory_qwen35/`
- Benchmark builders: `sft_grpo_training_evaluation/benchmark_tools/`
- Data interfaces: `sft_data/`, `sft_image_data/`, `sft_merged/`, `grpodata/`, `verl_data/`, and `benchmark/`
- Main flow: merge SFT inputs, convert reasoning data for verl, run GRPO, and evaluate a model.

Create the input and output directories required by the selected workflow before running its entry points.
