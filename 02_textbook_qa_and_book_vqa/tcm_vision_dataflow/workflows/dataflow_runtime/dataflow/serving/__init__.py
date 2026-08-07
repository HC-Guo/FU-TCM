from .api_llm_serving_request import APILLMServing_request

# 其余 serving 都可能依赖可选重依赖（openai/google/torch/transformers/vllm/sglang 等），统一做可选导入，
# 以便在最小环境下也能 import dataflow.serving 并使用 APILLMServing_request。
APIVLMServing_openai = None
PerspectiveAPIServing = None
LiteLLMServing = None
LocalHostLLMAPIServing_vllm = None
LocalModelLALMServing_vllm = None
LocalEmbeddingServing = None
LightRAGServing = None
APIGoogleVertexAIServing = None

LocalModelLLMServing_vllm = None
LocalModelLLMServing_sglang = None
LocalVLMServing_vllm = None

# openai / vlm
try:  # pragma: no cover
    from .api_vlm_serving_openai import APIVLMServing_openai
except Exception:
    pass

# google
try:  # pragma: no cover
    from .google_api_serving import PerspectiveAPIServing
except Exception:
    pass

try:  # pragma: no cover
    from .api_google_vertexai_serving import APIGoogleVertexAIServing
except Exception:
    pass

# litellm
try:  # pragma: no cover
    from .lite_llm_serving import LiteLLMServing
except Exception:
    pass

# local host api / local model
try:  # pragma: no cover
    from .localhost_llm_api_serving import LocalHostLLMAPIServing_vllm
except Exception:
    pass

try:  # pragma: no cover
    from .localmodel_lalm_serving import LocalModelLALMServing_vllm
except Exception:
    pass

try:  # pragma: no cover
    from .LocalSentenceLLMServing import LocalEmbeddingServing
except Exception:
    pass

try:  # pragma: no cover
    from .light_rag_serving import LightRAGServing
except Exception:
    pass

# 这些 serving 依赖可选的重依赖（torch/transformers/huggingface_hub/vllm/sglang 等）。
# 为了让最小环境也能 import dataflow.serving（跑基础单测/只用 API serving），这里做可选导入。
try:  # pragma: no cover
    from .local_model_llm_serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
except Exception:
    pass

try:  # pragma: no cover
    from .local_model_vlm_serving import LocalVLMServing_vllm
except Exception:
    pass


__all__ = [
    "APIGoogleVertexAIServing",
    "APILLMServing_request",
    "APIVLMServing_openai",
    "PerspectiveAPIServing",
    "LiteLLMServing",
    "LocalModelLALMServing_vllm",
    "LocalHostLLMAPIServing_vllm",
    # 可选项：若依赖缺失，以上符号会是 None
    "LocalModelLLMServing_vllm",
    "LocalModelLLMServing_sglang",
    "LocalVLMServing_vllm",
]
