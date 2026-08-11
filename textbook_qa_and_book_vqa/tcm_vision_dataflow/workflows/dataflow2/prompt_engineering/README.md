# TCM Visual-QA Prompt Engineering

This directory separates prompt design by visual task so that facial inspection, pulse diagnosis, tongue diagnosis, and mixed inspection materials do not share incompatible extraction rules.

## Structure

- `common_design_rules.md`: shared extraction and grounding constraints.
- `mianzhen/`: facial-inspection prompts.
- `maizhen/`: pulse-diagnosis prompts.
- `shezhen/`: tongue-diagnosis prompts.
- `wangzhen_mixed/`: mixed or not-yet-classified inspection sources.

## Design goals

- Produce one grounded QA record for one image.
- Permit cross-page matching when an image and its explanation are separated.
- Return an empty result when the source lacks enough evidence.
- Keep output fields compatible with the existing VQA generators.
- Route each source to task-specific prompts, image filters, and post-processing checks.
