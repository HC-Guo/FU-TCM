# FU-TCM model overview

FU-TCM is a multimodal large-language-model family for Traditional Chinese Medicine. It is designed to make the path from four-examination evidence through intermediate *bianzheng* judgments to the final syndrome explicit and reviewable.

## Model family

| Model | Base model | Modalities | Training objective |
| --- | --- | --- | --- |
| FU-TCM-9B | Qwen3.5-9B | Text and image | Domain SFT, cold-start SFT, and BGPO |
| FU-TCM-27B | Qwen3.6-27B | Text and image | Domain SFT, cold-start SFT, and BGPO |

The public repository currently contains the executable Qwen3.5-9B SFT, policy-optimization, inference, and evaluation path.

## Core capabilities

### TCM knowledge

FU-TCM is trained to answer questions grounded in classical works, formulas, materia medica, medical theory, textbooks, and clinical knowledge.

### Multimodal understanding

Visual instruction tuning covers illustrated medical materials and TCM images, including tongue, face, herb, and related diagnostic content.

### Structured syndrome differentiation

FU-TCM organizes case reasoning around:

- evidence from inspection, listening and smelling, inquiry, and palpation;
- the eight principles of exterior/interior, cold/heat, deficiency/excess, and yin/yang;
- organ-system judgments;
- qi, blood, and body-fluid patterns;
- wind, cold, summer heat, dampness, dryness, and fire;
- the final syndrome prediction.

Clinical cases contain 30 intermediate fields that connect the evidence to the final syndrome. These fields make the diagnostic path available for evaluation rather than leaving it as an unobserved internal step.

## BGPO alignment objective

Bianzheng-Grounded Policy Optimization is a TCM-specific instantiation of group-relative policy optimization. Its deterministic rewards evaluate:

- compliance with the required output structure;
- consistency between predicted and reference syndromes;
- fidelity across the intermediate *bianzheng* fields.

The syndrome-consistency term can assign graded credit through a standardized syndrome taxonomy rather than treating every non-exact prediction equally.

## Evaluation summary

FU-TCM-27B achieved an 80.41% macro-average across six benchmarks and ranked first on five of the six tasks. Across eight models, strict intermediate-parameter accuracy correlated with final syndrome-choice accuracy at Pearson *r* = 0.81.

These results are based on retrospective benchmarks. They do not establish clinical benefit or autonomous diagnostic safety.

See the [training overview](training_overview.md) for the executable stages.
