import torch

from llm_hookkit import (
    MemoryCost,
    TrainTemporaryFFNConfig,
    build_exact_attention_ffn,
    fit_temporary_ffn,
    reconstruction_metrics,
    scaled_dot_product_attention,
)


def test_exact_attention_ffn_matches_full_attention():
    torch.manual_seed(0)
    q = torch.randn(5, 4)
    k = torch.randn(7, 4)
    v = torch.randn(7, 4)

    teacher = scaled_dot_product_attention(q, k, v)
    student = build_exact_attention_ffn(k, v)

    assert torch.allclose(student(q), teacher, atol=1e-6)


def test_memory_cost_uses_expected_parameter_counts():
    cost = MemoryCost(context_length=1024, memory_units=128, layers=12, heads=8, head_dim=64)

    assert cost.kv_cache_parameters == 2 * 1024 * 12 * 8 * 64
    assert cost.temp_ffn_parameters == 2 * 128 * 12 * 8 * 64
    assert cost.compression_ratio == 8


def test_fit_temporary_ffn_reconstructs_easy_exact_case():
    torch.manual_seed(1)
    q = torch.randn(16, 3)
    k = torch.randn(4, 3)
    v = torch.randn(4, 3)
    target = scaled_dot_product_attention(q, k, v)

    _, metrics, _ = fit_temporary_ffn(
        q,
        target,
        TrainTemporaryFFNConfig(
            memory_units=4,
            steps=1,
            lr=1e-3,
            cosine_weight=0.0,
            init="sample",
            seed=1,
        ),
        k_init=k,
        v_init=v,
    )
    exact_metrics = reconstruction_metrics(target, build_exact_attention_ffn(k, v)(q))

    assert exact_metrics["mse"] < 1e-12
    assert metrics["mse"] < 1.0
