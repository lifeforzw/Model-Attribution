"""
Minimal attribution examples for a preloaded PyTorch model.

The important design choice is score_fn: it maps the model output to the
single scalar target you want to explain, such as one token logit.
"""

from llm_hookkit import (
    ActivationLocation,
    AttributionPatchingRunner,
    IntegratedGradientsRunner,
    ModelCall,
    NameMatcher,
)


def run_integrated_gradients(model, batch, target_token_id):
    def score_fn(output):
        logits = output.logits if hasattr(output, "logits") else output
        return logits[:, -1, target_token_id].sum()

    location = ActivationLocation(
        matcher=NameMatcher(r".*mlp.*"),
        attribution_slice=(slice(None), -1, slice(None)),
    )
    return IntegratedGradientsRunner(
        model=model,
        location=location,
        score_fn=score_fn,
        steps=32,
    ).run(ModelCall(kwargs=batch))


def run_attribution_patching(model, clean_batch, corrupt_batch, target_token_id):
    def score_fn(output):
        logits = output.logits if hasattr(output, "logits") else output
        return logits[:, -1, target_token_id].sum()

    location = ActivationLocation(
        matcher=NameMatcher(r".*mlp.*"),
        attribution_slice=(slice(None), -1, slice(None)),
    )
    return AttributionPatchingRunner(
        model=model,
        location=location,
        score_fn=score_fn,
    ).run(
        clean_inputs=ModelCall(kwargs=clean_batch),
        corrupt_inputs=ModelCall(kwargs=corrupt_batch),
    )
