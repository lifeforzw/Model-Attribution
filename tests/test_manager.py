from llm_hookkit import HookManager, HookSpec, HookType, NameMatcher, ScaleTransform, make_activation_callback
from llm_hookkit.config import config_from_args, parse_config, specs_from_config


class DummyHandle:
    def __init__(self, bucket, callback):
        self.bucket = bucket
        self.callback = callback
        self.removed = False

    def remove(self):
        self.removed = True
        self.bucket.remove(self.callback)


class DummyModule:
    def __init__(self, name):
        self.name = name
        self.forward_hooks = []
        self.forward_pre_hooks = []

    def register_forward_hook(self, callback):
        self.forward_hooks.append(callback)
        return DummyHandle(self.forward_hooks, callback)

    def register_forward_pre_hook(self, callback):
        self.forward_pre_hooks.append(callback)
        return DummyHandle(self.forward_pre_hooks, callback)

    def register_full_backward_hook(self, callback):
        return DummyHandle([], callback)

    def __call__(self, value):
        inputs = (value,)
        for hook in list(self.forward_pre_hooks):
            maybe_inputs = hook(self, inputs)
            if maybe_inputs is not None:
                inputs = maybe_inputs

        output = inputs[0] + 1
        for hook in list(self.forward_hooks):
            maybe_output = hook(self, inputs, output)
            if maybe_output is not None:
                output = maybe_output
        return output


class DummyModel:
    def __init__(self):
        self.layers = [DummyModule("layers.0"), DummyModule("layers.1")]

    def named_modules(self):
        yield "", self
        for index, layer in enumerate(self.layers):
            yield f"layers.{index}", layer


def test_forward_hook_can_modify_output():
    model = DummyModel()
    manager = HookManager.for_model(model)

    spec = HookSpec(
        name="double_first_layer",
        hook_type=HookType.FORWARD,
        matcher=NameMatcher(r"layers\.0"),
        callback=lambda ctx, module, inputs, output: output * 2,
    )

    with manager.apply(spec):
        assert model.layers[0](10) == 22
        assert model.layers[1](10) == 11

    assert model.layers[0](10) == 11


def test_forward_pre_hook_can_modify_inputs():
    model = DummyModel()
    manager = HookManager.for_model(model)

    spec = HookSpec(
        name="shift_input",
        hook_type=HookType.FORWARD_PRE,
        matcher=NameMatcher(r"layers\.1"),
        callback=lambda ctx, module, inputs, output=None: (inputs[0] + 5,),
    )

    with manager.apply(spec):
        assert model.layers[1](10) == 16


def test_hook_context_shares_state_across_modules():
    model = DummyModel()
    manager = HookManager.for_model(model)
    seen = []

    def record(ctx, module, inputs, output):
        ctx.state.setdefault("count", 0)
        ctx.state["count"] += 1
        seen.append((ctx.module_name, ctx.state["count"]))
        return output

    spec = HookSpec(
        name="record_all",
        hook_type=HookType.FORWARD,
        matcher=NameMatcher(r"layers"),
        callback=record,
    )

    with manager.apply(spec):
        model.layers[0](1)
        model.layers[1](1)

    assert seen == [("layers.0", 1), ("layers.1", 2)]


def test_activation_transform_interface_modifies_forward_output():
    model = DummyModel()
    manager = HookManager.for_model(model)

    spec = HookSpec(
        name="scale_output",
        hook_type=HookType.FORWARD,
        matcher=NameMatcher(r"layers\.0"),
        callback=make_activation_callback(ScaleTransform(factor=3)),
    )

    with manager.apply(spec):
        assert model.layers[0](2) == 9


def test_config_builds_specs_with_transform():
    model = DummyModel()
    config = parse_config(
        {
            "model": {"provider": "transformers", "name_or_path": "gpt2"},
            "hooks": [
                {
                    "name": "scale_layer",
                    "type": "forward",
                    "matcher": {"kind": "name", "pattern": "layers\\.1"},
                    "transform": {"kind": "scale", "params": {"factor": 4}},
                }
            ],
        }
    )

    specs = specs_from_config(config)
    with HookManager.for_model(model).apply(specs):
        assert model.layers[1](2) == 12
