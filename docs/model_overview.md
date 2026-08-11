# FU-TCM model overview

FU-TCM is a multimodal large language model for Traditional Chinese Medicine. The current training configuration adapts Qwen3.5-9B to TCM knowledge understanding, visual diagnosis, clinical reasoning, and structured syndrome differentiation.

## Model configuration

| Item | Configuration |
| --- | --- |
| Base model | Qwen3.5-9B |
| Modalities | Text and image |
| Supervised alignment | Full-parameter SFT |
| Reasoning alignment | verl GRPO |
| SFT context length | 4,096 tokens |
| Training precision | bf16 |
| Distributed training | DeepSpeed ZeRO-3 |

## Core capabilities

### TCM knowledge

The model is trained to answer questions grounded in classical texts, formulas, materia medica, medical theory, textbooks, and clinical knowledge.

### Multimodal understanding

Visual instruction tuning covers TCM images and illustrated medical materials, including tongue, face, herb, and related diagnostic content.

### Structured syndrome differentiation

FU-TCM organizes clinical reasoning around:

- evidence from inspection, listening and smelling, inquiry, and palpation;
- the eight principles of exterior/interior, cold/heat, deficiency/excess, and yin/yang;
- organ-system judgments;
- qi, blood, and body-fluid patterns;
- wind, cold, summer heat, dampness, dryness, and fire;
- final syndrome prediction.

## Alignment objective

The GRPO reward combines three signals:

- output and reasoning format: 20%;
- syndrome agreement: 40%;
- structured differentiation consistency: 40%.

This design rewards the reasoning structure and the final syndrome result rather than surface fluency alone.

See [Training overview](training_overview.md) for the executable training stages.
