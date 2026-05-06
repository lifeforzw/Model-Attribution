"""
Example for a preloaded HuggingFace/PyTorch model.

Install optional dependencies first:

    pip install "llm-hookkit[torch]" transformers
"""

from llm_hookkit import HookManager, HookSpec, HookType, NameMatcher, ScaleTransform, make_activation_callback


def attach_attention_logger(model):
    manager = HookManager.for_model(model)

    def log_attention_blocks(ctx, module, inputs, output):
        print(f"[{ctx.spec_name}] {ctx.module_name}: {type(module).__name__}")
        return output

    return manager.register(
        HookSpec(
            name="attention_logger",
            hook_type=HookType.FORWARD,
            matcher=NameMatcher(r"(attn|attention)"),
            callback=log_attention_blocks,
        )
    )


def attach_attention_scaler(model, factor=0.5):
    manager = HookManager.for_model(model)
    return manager.register(
        HookSpec(
            name="attention_scaler",
            hook_type=HookType.FORWARD,
            matcher=NameMatcher(r"(attn|attention)"),
            callback=make_activation_callback(ScaleTransform(factor=factor)),
        )
    )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    handles = attach_attention_logger(model)
    try:
        inputs = tokenizer("Hello hook framework", return_tensors="pt")
        model(**inputs)
    finally:
        handles.remove()
