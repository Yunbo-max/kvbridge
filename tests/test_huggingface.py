from types import SimpleNamespace

import torch

from kvbridge.huggingface import (
    _canonical_query,
    _decoder_layers,
    _apply_query_position_scaling,
    _rotary_module,
    model_signature,
    tokenizer_fingerprint,
)


class FakeTokenizer:
    special_tokens_map = {"eos_token": "</s>"}

    def get_vocab(self):
        return {"hello": 1, "world": 2}

    def get_added_vocab(self):
        return {"</s>": 3}


def test_tokenizer_fingerprint_is_deterministic() -> None:
    assert tokenizer_fingerprint(FakeTokenizer()) == tokenizer_fingerprint(FakeTokenizer())


def test_canonical_query_handles_normalized_and_projected_layouts() -> None:
    normalized = torch.arange(2 * 5 * 4 * 8).reshape(2, 5, 4, 8)
    projected = normalized.reshape(2, 5, 32)
    expected = normalized.permute(0, 2, 1, 3)

    torch.testing.assert_close(
        _canonical_query(normalized, query_heads=4, head_dim=8), expected
    )
    torch.testing.assert_close(
        _canonical_query(projected, query_heads=4, head_dim=8), expected
    )


def test_multimodal_wrapper_uses_text_decoder_contract() -> None:
    rotary = object()
    layers = [object(), object()]
    text_config = SimpleNamespace(
        head_dim=8,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        model_type="fake_text",
    )
    outer_config = SimpleNamespace(
        _name_or_path="fake/multimodal",
        architectures=["FakeConditionalGeneration"],
        text_config=text_config,
    )
    language_model = SimpleNamespace(rotary_emb=rotary, layers=layers)
    model = SimpleNamespace(
        config=outer_config,
        model=SimpleNamespace(language_model=language_model),
    )

    signature = model_signature(model, FakeTokenizer(), revision="pinned")
    assert signature.num_layers == 2
    assert signature.num_kv_heads == 2
    assert signature.head_dim == 8
    assert signature.architecture == "FakeConditionalGeneration"
    assert _rotary_module(model) is rotary
    assert _decoder_layers(model) is layers


def test_ministral_query_scaling_is_identity_then_logarithmic() -> None:
    query = torch.ones((1, 2, 3, 4))
    positions = torch.tensor([[0, 16, 32]])
    config = SimpleNamespace(
        rope_parameters={
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16,
        }
    )
    scaled = _apply_query_position_scaling(query, positions, config)
    expected = 1 + 0.1 * torch.log(torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(scaled[0, 0, :, 0], expected)
