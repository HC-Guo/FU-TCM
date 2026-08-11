"""Process-local compatibility aliases for verl without patching dependencies."""

try:
    import transformers
except Exception:
    transformers = None


if transformers is not None and not hasattr(transformers, "AutoModelForVision2Seq"):
    compatible_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if compatible_class is None:
        try:
            from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
        except Exception:
            compatible_class = None
        else:
            compatible_class = AutoModelForImageTextToText
    if compatible_class is not None:
        transformers.AutoModelForVision2Seq = compatible_class
