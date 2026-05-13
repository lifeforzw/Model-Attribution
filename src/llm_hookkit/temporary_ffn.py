from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TrainTemporaryFFNConfig:
    memory_units: int
    steps: int = 500
    lr: float = 1e-2
    batch_size: int | None = None
    cosine_weight: float = 0.1
    seed: int | None = None
    log_every: int = 50
    init: str = "sample"


@dataclass(frozen=True)
class MemoryCost:
    context_length: int
    memory_units: int
    layers: int = 1
    heads: int = 1
    head_dim: int = 64

    @property
    def kv_cache_parameters(self) -> int:
        return 2 * self.context_length * self.layers * self.heads * self.head_dim

    @property
    def temp_ffn_parameters(self) -> int:
        return 2 * self.memory_units * self.layers * self.heads * self.head_dim

    @property
    def compression_ratio(self) -> float:
        return self.context_length / self.memory_units

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "memory_units": self.memory_units,
            "layers": self.layers,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "kv_cache_parameters": self.kv_cache_parameters,
            "temp_ffn_parameters": self.temp_ffn_parameters,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class TemporaryFFNFitResult:
    memory_units: int
    metrics: dict[str, float]
    eval_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    memory_cost: MemoryCost | None = None
    model_state: dict[str, Any] | None = None

    def to_dict(self, *, include_state: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "memory_units": self.memory_units,
            "metrics": self.metrics,
            "eval_metrics": self.eval_metrics,
            "history": self.history,
            "memory_cost": self.memory_cost.to_dict() if self.memory_cost else None,
        }
        if include_state:
            payload["model_state"] = self.model_state
        return payload


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Temporary FFN experiments require PyTorch.") from exc
    return torch


def scaled_dot_product_attention(q, k, v):
    torch = require_torch()
    scale = math.sqrt(q.shape[-1])
    weights = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
    return weights @ v


class TemporarySoftmaxFFN:
    """Torch module wrapper created lazily to keep torch an optional dependency."""

    def __new__(cls, memory_units: int, head_dim: int):
        torch = require_torch()

        class _TemporarySoftmaxFFN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.memory_keys = torch.nn.Parameter(torch.empty(memory_units, head_dim))
                self.memory_values = torch.nn.Parameter(torch.empty(memory_units, head_dim))
                torch.nn.init.normal_(self.memory_keys, std=1.0 / math.sqrt(head_dim))
                torch.nn.init.normal_(self.memory_values, std=1.0 / math.sqrt(head_dim))

            def forward(self, q):
                weights = torch.softmax(q @ self.memory_keys.transpose(0, 1), dim=-1)
                return weights @ self.memory_values

        return _TemporarySoftmaxFFN()


def build_exact_attention_ffn(k, v):
    model = TemporarySoftmaxFFN(memory_units=k.shape[0], head_dim=k.shape[1])
    with require_torch().no_grad():
        model.memory_keys.copy_(k / math.sqrt(k.shape[-1]))
        model.memory_values.copy_(v)
    return model


def generate_random_qkv(
    *,
    context_length: int,
    head_dim: int,
    query_count: int,
    seed: int | None = None,
    device: str | None = None,
    dtype: str = "float32",
):
    torch = require_torch()
    if seed is not None:
        torch.manual_seed(seed)
    resolved_dtype = getattr(torch, dtype)
    kwargs = {"dtype": resolved_dtype}
    if device is not None:
        kwargs["device"] = device
    k = torch.randn(context_length, head_dim, **kwargs)
    v = torch.randn(context_length, head_dim, **kwargs)
    q = torch.randn(query_count, head_dim, **kwargs)
    return q, k, v


def fit_temporary_ffn(q, target, config: TrainTemporaryFFNConfig, *, k_init=None, v_init=None):
    torch = require_torch()
    if config.seed is not None:
        torch.manual_seed(config.seed)
    if q.ndim != 2 or target.ndim != 2:
        raise ValueError("q and target must be rank-2 tensors shaped [items, head_dim].")
    if q.shape[0] != target.shape[0] or q.shape[1] != target.shape[1]:
        raise ValueError("q and target must have matching item count and head dimension.")

    model = TemporarySoftmaxFFN(config.memory_units, q.shape[-1]).to(device=q.device, dtype=q.dtype)
    _initialize_student(model, config, k_init=k_init, v_init=v_init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    batch_size = config.batch_size or q.shape[0]
    history: list[dict[str, float]] = []

    for step in range(1, config.steps + 1):
        indices = _sample_indices(q.shape[0], batch_size, device=q.device)
        q_batch = q.index_select(0, indices)
        target_batch = target.index_select(0, indices)

        pred = model(q_batch)
        mse = torch.mean((pred - target_batch) ** 2)
        loss = mse
        cosine_loss = torch.zeros((), device=q.device, dtype=q.dtype)
        if config.cosine_weight:
            cosine_loss = 1 - torch.nn.functional.cosine_similarity(pred, target_batch, dim=-1).mean()
            loss = loss + config.cosine_weight * cosine_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == config.steps or (config.log_every and step % config.log_every == 0):
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "cosine_loss": float(cosine_loss.detach().cpu()),
                }
            )

    with torch.no_grad():
        prediction = model(q)
        metrics = reconstruction_metrics(target, prediction)

    return model, metrics, history


def run_compression_sweep(
    *,
    q,
    k,
    v,
    memory_units: Sequence[int],
    eval_q=None,
    steps: int = 500,
    lr: float = 1e-2,
    batch_size: int | None = None,
    cosine_weight: float = 0.1,
    seed: int | None = None,
    init: str = "sample",
    layers: int = 1,
    heads: int = 1,
) -> list[TemporaryFFNFitResult]:
    torch = require_torch()
    with torch.no_grad():
        target = scaled_dot_product_attention(q, k, v)
        eval_target = scaled_dot_product_attention(eval_q, k, v) if eval_q is not None else None

    results: list[TemporaryFFNFitResult] = []
    for index, units in enumerate(memory_units):
        config = TrainTemporaryFFNConfig(
            memory_units=units,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            cosine_weight=cosine_weight,
            seed=None if seed is None else seed + index,
            init=init,
        )
        model, metrics, history = fit_temporary_ffn(q, target, config, k_init=k, v_init=v)
        eval_metrics = None
        if eval_q is not None and eval_target is not None:
            with torch.no_grad():
                eval_prediction = model(eval_q)
                eval_metrics = reconstruction_metrics(eval_target, eval_prediction)
        results.append(
            TemporaryFFNFitResult(
                memory_units=units,
                metrics=metrics,
                eval_metrics=eval_metrics,
                history=history,
                memory_cost=MemoryCost(
                    context_length=k.shape[0],
                    memory_units=units,
                    layers=layers,
                    heads=heads,
                    head_dim=k.shape[-1],
                ),
                model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            )
        )
    return results


def reconstruction_metrics(target, prediction) -> dict[str, float]:
    torch = require_torch()
    diff = prediction - target
    mse = torch.mean(diff**2)
    rel = torch.linalg.vector_norm(diff) / torch.clamp_min(torch.linalg.vector_norm(target), 1e-12)
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
    return {
        "mse": float(mse.detach().cpu()),
        "relative_error": float(rel.detach().cpu()),
        "cosine_similarity": float(cosine.detach().cpu()),
    }


def summarize_results(results: Iterable[TemporaryFFNFitResult]) -> list[dict[str, Any]]:
    return [result.to_dict(include_state=False) for result in results]


def _initialize_student(model, config: TrainTemporaryFFNConfig, *, k_init=None, v_init=None) -> None:
    torch = require_torch()
    if config.init == "random" or k_init is None or v_init is None:
        return
    if config.init != "sample":
        raise ValueError("config.init must be 'sample' or 'random'.")
    if k_init.shape[0] < config.memory_units or v_init.shape[0] < config.memory_units:
        return
    indices = torch.linspace(
        0,
        k_init.shape[0] - 1,
        config.memory_units,
        device=k_init.device,
    ).round().long()
    with torch.no_grad():
        model.memory_keys.copy_(k_init.index_select(0, indices) / math.sqrt(k_init.shape[-1]))
        model.memory_values.copy_(v_init.index_select(0, indices))


def _sample_indices(item_count: int, batch_size: int, *, device):
    torch = require_torch()
    if batch_size >= item_count:
        return torch.arange(item_count, device=device)
    return torch.randint(0, item_count, (batch_size,), device=device)
