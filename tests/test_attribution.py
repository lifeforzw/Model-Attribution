import torch

from llm_hookkit import (
    ActivationLocation,
    AttributionPatchingRunner,
    IntegratedGradientsRunner,
    ModelCall,
    NameMatcher,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.out = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.out.weight.copy_(torch.tensor([[2.0, -3.0]]))

    def forward(self, x):
        hidden = self.proj(x)
        return self.out(hidden)


def score_fn(output):
    return output.sum()


def test_integrated_gradients_for_linear_activation_matches_exact_result():
    model = TinyModel()
    location = ActivationLocation(
        matcher=NameMatcher(r"proj"),
        attribution_slice=(0, slice(None)),
    )
    runner = IntegratedGradientsRunner(
        model=model,
        location=location,
        score_fn=score_fn,
        steps=8,
    )

    result = runner.run(ModelCall(args=(torch.tensor([[4.0, 5.0]]),)))

    assert torch.allclose(result.values["proj"], torch.tensor([8.0, -15.0]))


def test_attribution_patching_uses_clean_gradient_times_activation_delta():
    model = TinyModel()
    location = ActivationLocation(
        matcher=NameMatcher(r"proj"),
        attribution_slice=(0, slice(None)),
    )
    runner = AttributionPatchingRunner(
        model=model,
        location=location,
        score_fn=score_fn,
    )

    result = runner.run(
        clean_inputs=ModelCall(args=(torch.tensor([[4.0, 5.0]]),)),
        corrupt_inputs=ModelCall(args=(torch.tensor([[1.0, 9.0]]),)),
    )

    assert torch.allclose(result.values["proj"], torch.tensor([6.0, 12.0]))
