import json
from pathlib import Path

import pytest
import torch

from kvbridge.cache import KVCache
from kvbridge.config import FitConfig, ModelSignature
from kvbridge.fit import CalibrationPair
from kvbridge.errors import ArtifactError
from kvbridge.fit import fit_mapper
from kvbridge.mapper import CrossModelKVMapper
from kvbridge.synthetic import cache_r2, fit_demo, make_problem


@pytest.fixture(scope="module")
def fitted():
    problem = make_problem(calibration_pairs=5, tokens=18)
    return problem, fit_demo(problem)


def test_fit_selects_predictive_layers_and_generalizes(fitted) -> None:
    problem, mapper = fitted
    assert mapper.selected_layers == [[2], [0]]

    mapped = mapper.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)

    assert cache_r2(mapped, problem.evaluation.target) > 0.9999


def test_artifact_round_trip(fitted, tmp_path: Path) -> None:
    problem, mapper = fitted
    artifact = mapper.save(tmp_path / "artifact")
    loaded = CrossModelKVMapper.load(artifact)

    mapped = loaded.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)

    assert loaded.selected_layers == mapper.selected_layers
    assert cache_r2(mapped, problem.evaluation.target) > 0.9999


def test_artifact_detects_tampering(fitted, tmp_path: Path) -> None:
    _, mapper = fitted
    artifact = mapper.save(tmp_path / "artifact")
    weights = artifact / "mapper.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(ArtifactError, match="checksum"):
        CrossModelKVMapper.load(artifact)


def test_bfloat16_artifact_is_smaller_and_loads(fitted, tmp_path: Path) -> None:
    problem, mapper = fitted
    fp32 = mapper.save(tmp_path / "fp32", storage_dtype="float32")
    bf16 = mapper.save(tmp_path / "bf16", storage_dtype="bfloat16")

    loaded = CrossModelKVMapper.load(bf16)
    mapped = loaded.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)

    assert loaded.storage_dtype == "bfloat16"
    assert (bf16 / "mapper.safetensors").stat().st_size < (
        fp32 / "mapper.safetensors"
    ).stat().st_size
    assert cache_r2(mapped, problem.evaluation.target) > 0.999


def test_mapper_can_be_made_resident_with_explicit_compute_dtype() -> None:
    problem = make_problem(seed=71)
    mapper = fit_demo(problem)

    returned = mapper.to("cpu", dtype=torch.float64)

    assert returned is mapper
    assert mapper.device.type == "cpu"
    assert mapper.storage_dtype == "torch.float64"
    mapped = mapper.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)
    assert cache_r2(mapped, problem.evaluation.target) > 0.999


def test_artifact_rejects_manifest_dtype_mismatch(fitted, tmp_path: Path) -> None:
    _, mapper = fitted
    artifact = mapper.save(tmp_path / "artifact", storage_dtype="bfloat16")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["storage_dtype"] = "float32"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="dtype"):
        CrossModelKVMapper.load(artifact)


def test_fit_honors_token_stride() -> None:
    problem = make_problem(calibration_pairs=4, tokens=15)
    mapper = fit_mapper(
        problem.calibration,
        problem.source,
        problem.target,
        FitConfig(top_k=1, token_stride=3),
    )

    mapped = mapper.map(problem.evaluation.source, target_rotary=problem.evaluation.target.rotary)
    assert cache_r2(mapped, problem.evaluation.target) > 0.999


@pytest.mark.parametrize(("source_heads", "target_heads"), [(2, 4), (4, 2)])
def test_fit_supports_unmatched_kv_heads(source_heads: int, target_heads: int) -> None:
    generator = torch.Generator().manual_seed(123 + source_heads)
    dim, tokens = 4, 48
    source = ModelSignature("source", "rev", "tok", 2, source_heads, dim)
    target = ModelSignature("target", "rev", "tok", 1, target_heads, dim)
    weight = torch.randn(source_heads * dim, target_heads * dim, generator=generator)
    pairs = []
    for _ in range(4):
        source_layers = [
            torch.randn(1, source_heads, tokens, dim, generator=generator)
            for _ in range(2)
        ]
        flat = source_layers[1].permute(0, 2, 1, 3).reshape(tokens, -1)
        target_tensor = (flat @ weight).reshape(1, tokens, target_heads, dim).permute(0, 2, 1, 3)
        pairs.append(CalibrationPair(
            KVCache(source_layers, source_layers, keys_are_content=True),
            KVCache([target_tensor], [target_tensor], keys_are_content=True),
        ))
    mapper = fit_mapper(
        pairs[:-1], source, target,
        FitConfig(top_k=1, content_space=False, require_matched_kv=False),
    )

    mapped = mapper.map(pairs[-1].source)

    assert mapper.selected_layers == [[1]]
    assert cache_r2(mapped, pairs[-1].target) > 0.999
