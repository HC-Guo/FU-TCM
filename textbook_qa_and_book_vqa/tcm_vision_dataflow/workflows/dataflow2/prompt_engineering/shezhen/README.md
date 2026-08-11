# Tongue-Diagnosis Prompt Module

This module routes structurally different sections of a tongue-diagnosis source to distinct prompt templates.

## Theory-section route

Use [`ch2_vqa_prompt.md`](./ch2_vqa_prompt.md) for sections that provide long-form theory about tongue color, shape, movement, coating, vessels, textures, or related findings. The prompt covers visual description, core pathogenesis, spatial relationships, associated symptoms, prognosis, and relevant biomedical explanations when the source provides them.

## Clinical-section route

Use [`ch3_vqa_prompt.md`](./ch3_vqa_prompt.md) for concise clinical sections that map tongue-body and coating findings to a syndrome, stage, specific indicator, or treatment information.

## Output schema

```json
{
  "question": "Grounded question text",
  "answer": "Answer supported by the source text",
  "type": "task_type"
}
```

Answers must remain grounded in the provided source context. The source books and generated outputs are local and excluded from Git.
