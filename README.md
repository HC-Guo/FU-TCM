<h1 align="center">Fu-TCM: a multimodal large language model that learns traditional Chinese medicine diagnostic reasoning</h1>

<p align="center">
  🤗 <a href="https://huggingface.co/fudanxai/Fu-TCM-9B">Fu-TCM-9B</a> |
  🤗 <a href="https://huggingface.co/fudanxai/Fu-TCM-27B">Fu-TCM-27B</a>
</p>

## Overview

FU-TCM is a multimodal large-language-model family for Traditional Chinese Medicine (TCM), trained on approximately **7.12 million examples** derived from classical texts, textbooks, medical images, clinical cases, and public datasets. It is designed to connect evidence from the four diagnostic methods to explicit intermediate *bianzheng* fields and a final syndrome prediction.

Training proceeds in two stages. **TCM domain-specific learning** uses text QA, image-text VQA, and case-reasoning examples to establish TCM knowledge, multimodal understanding, and structured reasoning. **Bianzheng-Grounded Policy Optimization (BGPO)** then optimizes response format, tree-based syndrome consistency, and fidelity across 30 intermediate *bianzheng* fields.

<p align="center">
  <a href="assets/fu-tcm-framework-overview.png">
    <img src="assets/fu-tcm-framework-overview.png" alt="Four-panel FU-TCM framework covering data construction, two-stage training, benchmark performance, and external validation" width="900">
  </a>
</p>

<p align="center"><em>FU-TCM framework: multisource data construction, two-stage training, benchmark performance, and external validation.</em></p>

## Key Features

- 🧠 **From knowledge to *bianzheng*.** Fu-TCM shifts TCM large-model training from knowledge accumulation toward learning how clinical evidence is organized into *bianzheng* decisions.
- 🔎 **Multimodal four-examination reasoning.** A unified architecture integrates evidence from the four diagnostic methods and generates a complete, traceable evidence-to-syndrome chain for clinician review.
- 📊 **Process–outcome evaluation.** A dual-dimensional evaluation system measures both final predictions and the intermediate *bianzheng* process. Fu-TCM achieves state-of-the-art results across six benchmarks and professional-level external validation.
- 🔓 **Computable and open TCM knowledge.** TCM *bianzheng* experience is encoded as 30 learnable fields and a syndrome taxonomy, while open data, code, and model weights lower the cost of reuse and accelerate international research.
