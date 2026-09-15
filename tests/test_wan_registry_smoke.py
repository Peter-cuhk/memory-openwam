"""Wan video backbone registry smoke tests.

Verifies that the three Wan backbones plan §1 promises are wired up
without touching GPU or real weights.
"""

from __future__ import annotations

import pytest


def test_registry_keys_present():
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY

    assert {"wan22_ti2v_5b", "wan21_vace_1_3b", "wan21_i2v_14b_480p"} <= set(_VIDEO_BACKBONE_REGISTRY)


def test_unknown_key_raises():
    from openwam.model.video_backbone import build_video_backbone

    with pytest.raises(KeyError, match="Unknown video backbone"):
        build_video_backbone("nonexistent_backbone", {})


def test_cached_training_skips_t5_and_tokenizer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from openwam.model.video_backbone.wan import encode, loader, pipeline_builder

    path = tmp_path / "embeddings.pt"
    entry = (torch.zeros(1, 512, 4096, dtype=torch.bfloat16), torch.tensor([2]))
    torch.save({"metadata": encode.text_cache_metadata(str(tmp_path)), "embeddings": {"task": entry}}, path)
    monkeypatch.setattr(
        pipeline_builder,
        "discover_model_files",
        lambda _: (
            [SimpleNamespace(path="models_t5_umt5-xxl-enc-bf16.pth"), SimpleNamespace(path="dit.safetensors")],
            "tokenizer",
        ),
    )

    def load_components(configs, tokenizer, **kw):
        assert [c.path for c in configs] == ["dit.safetensors"]
        assert tokenizer is None
        return SimpleNamespace()

    monkeypatch.setattr(loader, "load_wan_components", load_components)
    loader.build_holder({"video_backbone": {"model_path": str(tmp_path), "skip_text_encoder": True}})
    cache = encode.load_text_cache(str(path), str(tmp_path))
    context, lengths = encode.lookup_text_cache(["task", "task"], cache, device="cpu", dtype=torch.bfloat16)
    assert context.shape == (2, 512, 4096)
    assert lengths.tolist() == [2, 2]
    with pytest.raises(KeyError, match="Prompt missing"):
        encode.lookup_text_cache(["unknown"], cache, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not match"):
        encode.load_text_cache(str(path), "different-model")
