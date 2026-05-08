import torch

from llm_hookkit import (
    ActivationLocation,
    NameMatcher,
    SRPropagationConfig,
    SRPropagationRunner,
    SeparatedPrompt,
    build_embedding_path_points,
    find_subject_token_indices,
)


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.block0 = torch.nn.Linear(2, 2, bias=False)
        self.block1 = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [3.0, 0.0],
                    ]
                )
            )
            self.block0.weight.copy_(torch.eye(2))
            self.block1.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden = self.block0(inputs_embeds)
        hidden = self.block1(hidden)
        return hidden


class TinyTokenizer:
    vocab = {"The": 10, " capital": 11, " of": 12, " France": 13, " is": 14, "France": 15}

    def encode(self, text, add_special_tokens=False):
        if text == "The capital of France is":
            return [10, 11, 12, 13, 14]
        if text == " France":
            return [13]
        if text == "France":
            return [15]
        return []


def test_find_subject_token_indices_requires_unique_token_span():
    indices = find_subject_token_indices(
        TinyTokenizer(),
        "The capital of France is",
        "France",
        [10, 11, 12, 13, 14],
    )

    assert indices == [3]


def test_embedding_path_points_zero_relation_and_subject_positions():
    full = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])

    points = build_embedding_path_points(
        full,
        subject_token_indices=[1],
        relation_token_indices=[0, 2],
        pairs=((0.0, 0.0, "baseline"), (1.0, 0.0, "subject"), (1.0, 1.0, "full")),
    )

    assert torch.allclose(points[0].inputs_embeds, torch.zeros_like(full))
    assert torch.allclose(points[1].inputs_embeds, torch.tensor([[[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]]))
    assert torch.allclose(points[2].inputs_embeds, full)


def test_sr_runner_computes_hand_checkable_deltas_and_metrics():
    model = TinyEmbeddingModel()
    prompt = SeparatedPrompt(
        prompt="S R",
        subject_text="S",
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        tokens=["S", "R", "R2"],
        subject_token_indices=[0],
        relation_token_indices=[1, 2],
    )
    location = ActivationLocation(matcher=NameMatcher(r"block0"), name="residual")
    config = SRPropagationConfig(
        locations=(location,),
        alphas=(0.0, 0.5, 1.0),
        betas=(0.0, 0.5, 1.0),
        pca_token_positions=(-1,),
    )

    result = SRPropagationRunner(model=model, config=config).run(prompt)

    subject_delta = result.values["subject"]["residual"]["block0"]
    relation_delta = result.values["relation"]["residual"]["block0"]
    assert torch.allclose(subject_delta, torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]))
    assert torch.allclose(relation_delta, torch.tensor([[[0.0, 0.0], [0.0, 2.0], [3.0, 0.0]]]))
    assert torch.allclose(
        result.metrics["norm_subject"]["residual"]["block0"],
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        result.metrics["norm_relation"]["residual"]["block0"],
        torch.tensor([[0.0, 2.0, 3.0]]),
    )
    assert result.metrics["subject_path"]["velocity_norm"]["residual"]["block0"].shape == (2, 1, 3)
    assert result.metrics["relation_path"]["acceleration_norm"]["residual"]["block0"].shape == (1, 1, 3)
