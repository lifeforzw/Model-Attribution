from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .attribution import ActivationLocation, ModelCall
from .manager import HookManager
from .spec import HookSpec


@dataclass(frozen=True)
class SeparatedPrompt:
    prompt: str
    subject_text: str
    input_ids: Any
    attention_mask: Any | None = None
    tokens: Sequence[str] = ()
    subject_token_indices: Sequence[int] = ()
    relation_token_indices: Sequence[int] = ()
    target_position: int = -1
    target_text: str | None = None
    target_token_id: int | None = None
    relation_text: str | None = None
    relation_template: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingPathPoint:
    name: str
    alpha: float
    beta: float
    inputs_embeds: Any
    attention_mask: Any | None = None

    def to_model_call(self) -> ModelCall:
        kwargs = {"inputs_embeds": self.inputs_embeds}
        if self.attention_mask is not None:
            kwargs["attention_mask"] = self.attention_mask
        return ModelCall(kwargs=kwargs)


@dataclass(frozen=True)
class SRPropagationConfig:
    locations: Sequence[ActivationLocation]
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    betas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    pca_token_positions: Sequence[int] = (-1,)
    principal_angle_rank: int = 4


@dataclass(frozen=True)
class SRPropagationResult:
    values: Mapping[str, Any]
    activations: Mapping[str, Any]
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SRPropagationRunner:
    model: Any
    config: SRPropagationConfig
    manager: HookManager | None = None

    def run(self, separated_prompt: SeparatedPrompt) -> SRPropagationResult:
        torch = _require_torch()
        manager = self.manager or HookManager.for_model(self.model)
        embedding_layer = _get_input_embedding_layer(self.model)
        input_ids = _ensure_batch(separated_prompt.input_ids, torch).to(_module_device(embedding_layer))
        attention_mask = separated_prompt.attention_mask
        if attention_mask is not None:
            attention_mask = _ensure_batch(attention_mask, torch).to(input_ids.device)

        with torch.no_grad():
            full_embeddings = embedding_layer(input_ids)
            core_points = build_embedding_path_points(
                full_embeddings,
                separated_prompt.subject_token_indices,
                separated_prompt.relation_token_indices,
                attention_mask=attention_mask,
                pairs=((0.0, 0.0, "baseline"), (1.0, 0.0, "subject"), (1.0, 1.0, "full")),
            )
            subject_points = build_embedding_path_points(
                full_embeddings,
                separated_prompt.subject_token_indices,
                separated_prompt.relation_token_indices,
                attention_mask=attention_mask,
                pairs=((alpha, 0.0, f"subject_path:{_format_coeff(alpha)}") for alpha in self.config.alphas),
            )
            relation_points = build_embedding_path_points(
                full_embeddings,
                separated_prompt.subject_token_indices,
                separated_prompt.relation_token_indices,
                attention_mask=attention_mask,
                pairs=((1.0, beta, f"relation_path:{_format_coeff(beta)}") for beta in self.config.betas),
            )

            core_activations = {
                point.name: capture_activations_for_locations(
                    self.model,
                    point.to_model_call(),
                    self.config.locations,
                    manager=manager,
                    detach=True,
                )
                for point in core_points
            }
            subject_trajectory = [
                capture_activations_for_locations(
                    self.model,
                    point.to_model_call(),
                    self.config.locations,
                    manager=manager,
                    detach=True,
                )
                for point in subject_points
            ]
            relation_trajectory = [
                capture_activations_for_locations(
                    self.model,
                    point.to_model_call(),
                    self.config.locations,
                    manager=manager,
                    detach=True,
                )
                for point in relation_points
            ]

        deltas = compute_sr_deltas(core_activations)
        metrics = compute_sr_metrics(
            deltas,
            subject_trajectory=subject_trajectory,
            relation_trajectory=relation_trajectory,
            alphas=self.config.alphas,
            betas=self.config.betas,
            pca_token_positions=_resolve_positions(
                self.config.pca_token_positions,
                sequence_length=full_embeddings.shape[1],
            ),
            principal_angle_rank=self.config.principal_angle_rank,
        )

        return SRPropagationResult(
            values=deltas,
            activations={
                "core": core_activations,
                "subject_path": subject_trajectory,
                "relation_path": relation_trajectory,
            },
            metrics=metrics,
            metadata={
                "method": "sr_separated_propagation",
                "prompt": prompt_metadata(separated_prompt),
                "alphas": [float(alpha) for alpha in self.config.alphas],
                "betas": [float(beta) for beta in self.config.betas],
                "locations": [location_metadata(location) for location in self.config.locations],
            },
        )


def separated_prompt_from_tokenizer(
    tokenizer: Any,
    *,
    prompt: str,
    subject_text: str,
    target_position: int = -1,
    target_text: str | None = None,
    target_token_id: int | None = None,
    relation_text: str | None = None,
    relation_template: str | None = None,
) -> SeparatedPrompt:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    token_ids = input_ids[0].tolist()
    subject_indices = find_subject_token_indices(tokenizer, prompt, subject_text, token_ids)
    relation_indices = [index for index in range(len(token_ids)) if index not in set(subject_indices)]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return SeparatedPrompt(
        prompt=prompt,
        subject_text=subject_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokens=tokens,
        subject_token_indices=subject_indices,
        relation_token_indices=relation_indices,
        target_position=target_position,
        target_text=target_text,
        target_token_id=target_token_id,
        relation_text=relation_text,
        relation_template=relation_template,
    )


def find_subject_token_indices(
    tokenizer: Any,
    prompt: str,
    subject_text: str,
    prompt_token_ids: Sequence[int] | None = None,
) -> list[int]:
    if prompt_token_ids is None:
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidates: list[list[int]] = []
    for prefix in ("", " "):
        subject_ids = tokenizer.encode(prefix + subject_text, add_special_tokens=False)
        if not subject_ids:
            continue
        candidates.extend(_find_subsequence(prompt_token_ids, subject_ids))
    unique = {tuple(candidate) for candidate in candidates}
    if len(unique) != 1:
        raise ValueError(
            f"Subject {subject_text!r} did not align to exactly one token span; found {len(unique)} spans."
        )
    return list(next(iter(unique)))


def build_embedding_path_points(
    full_embeddings: Any,
    subject_token_indices: Sequence[int],
    relation_token_indices: Sequence[int],
    *,
    attention_mask: Any | None = None,
    pairs: Sequence[tuple[float, float, str]] | Any,
) -> list[EmbeddingPathPoint]:
    torch = _require_torch()
    subject_mask = _token_mask(full_embeddings, subject_token_indices, torch)
    relation_mask = _token_mask(full_embeddings, relation_token_indices, torch)
    points = []
    for alpha, beta, name in pairs:
        embeddings = full_embeddings * (float(alpha) * subject_mask + float(beta) * relation_mask)
        points.append(
            EmbeddingPathPoint(
                name=name,
                alpha=float(alpha),
                beta=float(beta),
                inputs_embeds=embeddings.detach().clone(),
                attention_mask=attention_mask,
            )
        )
    return points


def capture_activations_for_locations(
    model: Any,
    inputs: ModelCall,
    locations: Sequence[ActivationLocation],
    *,
    manager: HookManager | None = None,
    detach: bool = True,
) -> dict[str, dict[str, Any]]:
    manager = manager or HookManager.for_model(model)
    captured: dict[str, dict[str, Any]] = {location.name: {} for location in locations}
    specs = []

    for location in locations:
        def callback(ctx, module, module_inputs, output=None, *, location=location):
            activation = _get_activation(output if output is not None else module_inputs, location.tensor_path)
            captured[location.name][ctx.module_name] = activation.detach().clone() if detach else activation
            return output

        specs.append(
            HookSpec(
                name=f"{location.name}:sr_capture",
                hook_type=location.hook_type,
                matcher=location.matcher,
                callback=callback,
            )
        )

    with manager.apply(specs):
        model(*inputs.args, **dict(inputs.kwargs))

    missing = [name for name, values in captured.items() if not values]
    if missing:
        raise RuntimeError(f"No activations were captured for locations: {', '.join(missing)}.")
    return captured


def compute_sr_deltas(core_activations: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    baseline = core_activations["baseline"]
    subject = core_activations["subject"]
    full = core_activations["full"]
    return {
        "subject": _map_activation_pairs(subject, baseline, lambda left, right: left - right),
        "relation": _map_activation_pairs(full, subject, lambda left, right: left - right),
    }


def compute_sr_metrics(
    deltas: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    subject_trajectory: Sequence[Mapping[str, Mapping[str, Any]]],
    relation_trajectory: Sequence[Mapping[str, Mapping[str, Any]]],
    alphas: Sequence[float],
    betas: Sequence[float],
    pca_token_positions: Sequence[int],
    principal_angle_rank: int = 4,
) -> dict[str, Any]:
    return {
        "norm_subject": _map_nested_tensors(deltas["subject"], _l2_last_dim),
        "norm_relation": _map_nested_tensors(deltas["relation"], _l2_last_dim),
        "cosine_subject_relation": _map_activation_pairs(
            deltas["subject"],
            deltas["relation"],
            _cosine_last_dim,
        ),
        "principal_angles": _map_activation_pairs(
            deltas["subject"],
            deltas["relation"],
            lambda left, right: _principal_angles(left, right, rank=principal_angle_rank),
        ),
        "subject_path": _trajectory_metrics(subject_trajectory, alphas),
        "relation_path": _trajectory_metrics(relation_trajectory, betas),
        "pca": {
            "subject_path": _pca_trajectory(subject_trajectory, pca_token_positions),
            "relation_path": _pca_trajectory(relation_trajectory, pca_token_positions),
        },
    }


def prompt_metadata(prompt: SeparatedPrompt) -> dict[str, Any]:
    input_ids = prompt.input_ids.detach().cpu().tolist() if hasattr(prompt.input_ids, "detach") else prompt.input_ids
    return {
        "prompt": prompt.prompt,
        "subject_text": prompt.subject_text,
        "relation_text": prompt.relation_text,
        "relation_template": prompt.relation_template,
        "input_ids": input_ids,
        "tokens": list(prompt.tokens),
        "subject_token_indices": list(prompt.subject_token_indices),
        "relation_token_indices": list(prompt.relation_token_indices),
        "target_position": prompt.target_position,
        "target_text": prompt.target_text,
        "target_token_id": prompt.target_token_id,
        "metadata": dict(prompt.metadata),
    }


def location_metadata(location: ActivationLocation) -> dict[str, Any]:
    return {
        "name": location.name,
        "hook_type": location.hook_type.value,
        "tensor_path": list(location.tensor_path),
        "attribution_slice": repr(location.attribution_slice),
        "matcher": repr(location.matcher),
    }


def _map_nested_tensors(nested: Mapping[str, Mapping[str, Any]], fn) -> dict[str, dict[str, Any]]:
    return {
        location_name: {module_name: fn(value) for module_name, value in modules.items()}
        for location_name, modules in nested.items()
    }


def _map_activation_pairs(left, right, fn):
    result = {}
    for location_name, left_modules in left.items():
        if location_name not in right:
            raise RuntimeError(f"Missing activation location '{location_name}'.")
        result[location_name] = {}
        for module_name, left_value in left_modules.items():
            if module_name not in right[location_name]:
                raise RuntimeError(f"Missing activation for '{location_name}:{module_name}'.")
            result[location_name][module_name] = fn(left_value, right[location_name][module_name])
    return result


def _trajectory_metrics(trajectory: Sequence[Mapping[str, Mapping[str, Any]]], coefficients: Sequence[float]):
    if len(trajectory) < 2:
        return {"velocity_norm": {}, "acceleration_norm": {}, "adjacent_velocity_cosine": {}}
    velocities = _finite_differences(trajectory, coefficients)
    accelerations = _finite_differences(velocities, _segment_midpoints(coefficients))
    return {
        "velocity_norm": _map_nested_tensors_from_sequence(velocities, _l2_last_dim),
        "acceleration_norm": _map_nested_tensors_from_sequence(accelerations, _l2_last_dim),
        "adjacent_velocity_cosine": _adjacent_cosines(velocities),
    }


def _finite_differences(sequence: Sequence[Mapping[str, Mapping[str, Any]]], coefficients: Sequence[float]):
    diffs = []
    for index in range(len(sequence) - 1):
        delta = float(coefficients[index + 1]) - float(coefficients[index])
        if delta == 0:
            raise ValueError("Interpolation coefficients must be distinct for finite differences.")
        diffs.append(
            _map_activation_pairs(
                sequence[index + 1],
                sequence[index],
                lambda left, right, delta=delta: (left - right) / delta,
            )
        )
    return diffs


def _map_nested_tensors_from_sequence(sequence, fn):
    if not sequence:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for location_name, modules in sequence[0].items():
        result[location_name] = {}
        for module_name in modules:
            stacked = _stack([item[location_name][module_name] for item in sequence])
            result[location_name][module_name] = fn(stacked)
    return result


def _adjacent_cosines(velocities):
    if len(velocities) < 2:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for location_name, modules in velocities[0].items():
        result[location_name] = {}
        for module_name in modules:
            values = []
            for index in range(len(velocities) - 1):
                values.append(
                    _cosine_last_dim(
                        velocities[index][location_name][module_name],
                        velocities[index + 1][location_name][module_name],
                    )
                )
            result[location_name][module_name] = _stack(values)
    return result


def _pca_trajectory(trajectory: Sequence[Mapping[str, Mapping[str, Any]]], token_positions: Sequence[int]):
    if not trajectory:
        return {}
    torch = _require_torch()
    result: dict[str, dict[str, dict[int, Any]]] = {}
    for location_name, modules in trajectory[0].items():
        result[location_name] = {}
        for module_name in modules:
            stacked = torch.stack([item[location_name][module_name] for item in trajectory], dim=0)
            result[location_name][module_name] = {}
            for position in token_positions:
                vectors = stacked[:, 0, position, :].float()
                centered = vectors - vectors.mean(dim=0, keepdim=True)
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                components = vh[: min(3, vh.shape[0])]
                result[location_name][module_name][int(position)] = centered @ components.T
    return result


def _principal_angles(left, right, *, rank: int):
    torch = _require_torch()
    left_samples = left.reshape(-1, left.shape[-1]).float()
    right_samples = right.reshape(-1, right.shape[-1]).float()
    left_basis = _orthonormal_basis(left_samples, rank)
    right_basis = _orthonormal_basis(right_samples, rank)
    singular_values = torch.linalg.svdvals(left_basis.T @ right_basis).clamp(-1.0, 1.0)
    angles = torch.arccos(singular_values)
    return {
        "angles": angles,
        "min_angle": angles.min() if angles.numel() else torch.tensor(float("nan"), device=left.device),
        "mean_angle": angles.mean() if angles.numel() else torch.tensor(float("nan"), device=left.device),
    }


def _orthonormal_basis(samples, rank: int):
    torch = _require_torch()
    centered = samples - samples.mean(dim=0, keepdim=True)
    if torch.linalg.vector_norm(centered) == 0:
        centered = samples
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    keep = int((singular_values > 1e-8).sum().item())
    keep = max(1, min(rank, keep, vh.shape[0]))
    return vh[:keep].T


def _l2_last_dim(value):
    torch = _require_torch()
    return torch.linalg.vector_norm(value.float(), dim=-1)


def _cosine_last_dim(left, right):
    torch = _require_torch()
    return torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-12)


def _stack(values):
    torch = _require_torch()
    return torch.stack(list(values), dim=0)


def _segment_midpoints(coefficients: Sequence[float]) -> list[float]:
    return [(float(coefficients[index]) + float(coefficients[index + 1])) / 2 for index in range(len(coefficients) - 1)]


def _resolve_positions(positions: Sequence[int], *, sequence_length: int) -> list[int]:
    return [position if position >= 0 else sequence_length + position for position in positions]


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> list[list[int]]:
    if len(subsequence) > len(sequence):
        return []
    matches = []
    for start in range(0, len(sequence) - len(subsequence) + 1):
        if list(sequence[start:start + len(subsequence)]) == list(subsequence):
            matches.append(list(range(start, start + len(subsequence))))
    return matches


def _token_mask(full_embeddings, token_indices: Sequence[int], torch):
    mask = torch.zeros(
        (1, full_embeddings.shape[1], 1),
        dtype=full_embeddings.dtype,
        device=full_embeddings.device,
    )
    if token_indices:
        mask[:, list(token_indices), :] = 1
    return mask


def _get_input_embedding_layer(model):
    if hasattr(model, "get_input_embeddings"):
        embedding_layer = model.get_input_embeddings()
        if embedding_layer is not None:
            return embedding_layer
    if hasattr(model, "embed"):
        return model.embed
    if hasattr(model, "embedding"):
        return model.embedding
    raise TypeError("Model must expose get_input_embeddings(), embed, or embedding for SR propagation.")


def _ensure_batch(value, torch):
    if not hasattr(value, "ndim"):
        value = torch.tensor(value)
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return _require_torch().device("cpu")


def _get_activation(container: Any, path: Sequence[int | str]) -> Any:
    current = container
    for key in path:
        current = current[key]
    return current


def _format_coeff(value: float) -> str:
    return f"{float(value):.6g}"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("S/R propagation experiments require PyTorch.") from exc
    return torch
