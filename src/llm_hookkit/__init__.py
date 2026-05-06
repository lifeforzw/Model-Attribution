from .context import HookContext
from .manager import HookManager
from .matchers import AndMatcher, AnyMatcher, NameMatcher, PredicateMatcher, TypeMatcher
from .spec import HookSpec, HookType
from .attribution import (
    ActivationLocation,
    AttributionPatchingRunner,
    AttributionResult,
    IntegratedGradientsRunner,
    ModelCall,
    capture_activations,
    capture_activations_and_gradients,
    run_with_patched_activations,
)
from .transforms import (
    ActivationTransform,
    ComposeTransform,
    NoOpTransform,
    ReplaceTransform,
    ScaleTransform,
    TransformRegistry,
    make_activation_callback,
)

__all__ = [
    "ActivationTransform",
    "ActivationLocation",
    "AndMatcher",
    "AnyMatcher",
    "AttributionPatchingRunner",
    "AttributionResult",
    "ComposeTransform",
    "HookContext",
    "HookManager",
    "HookSpec",
    "HookType",
    "IntegratedGradientsRunner",
    "ModelCall",
    "NameMatcher",
    "NoOpTransform",
    "PredicateMatcher",
    "ReplaceTransform",
    "ScaleTransform",
    "TransformRegistry",
    "TypeMatcher",
    "capture_activations",
    "capture_activations_and_gradients",
    "make_activation_callback",
    "run_with_patched_activations",
]
