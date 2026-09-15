"""BaseWAMArchitecture.save_assets_for_deployment dispatches save_deploy_assets
to every backbone unconditionally — each backbone base declares the hook
(default no-op), so there is no hasattr probing.

Lets different backbones (Wan component specs + tokenizer, future ones) ship
their own deploy assets without the trainer importing them directly.
"""

from __future__ import annotations

import torch.nn as nn

from openwam.model.architectures.base import BaseWAMArchitecture


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def save_deploy_assets(self, output_dir, cfg):
        self.calls.append((output_dir, cfg))


class _ConcreteArch(BaseWAMArchitecture):
    """Concrete BaseWAMArchitecture so we can exercise the dispatcher in isolation."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError


def test_dispatch_to_all_backbones(tmp_path):
    arch = _ConcreteArch(cfg=None)
    vb = _RecordingBackbone()
    ab = _RecordingBackbone()
    arch.video_backbone = vb
    arch.action_backbone = ab

    cfg = {"model": {"video_backbone": {"model_path": str(tmp_path)}}}
    arch.save_assets_for_deployment(str(tmp_path), cfg)

    assert vb.calls == [(str(tmp_path), cfg)]
    assert ab.calls == [(str(tmp_path), cfg)]


def test_all_backbone_bases_declare_hook():
    """Direction-A contract: every backbone base declares save_deploy_assets on
    itself (default no-op), so the dispatcher calls it unconditionally rather than
    probing with hasattr. ``vars`` (not hasattr) catches a root that forgot it."""
    from openwam.model.action_backbone.base import ActionDiTBackbone, SharedActionBackbone
    from openwam.model.video_backbone.base import VideoBackbone
    from openwam.model.vlm_backbone.base import VlmBackbone

    for base in (VideoBackbone, SharedActionBackbone, ActionDiTBackbone, VlmBackbone):
        assert "save_deploy_assets" in vars(base), f"{base.__name__} must declare the no-op hook"


def test_wan_cached_text_assets_exclude_t5(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.wan import component_specs

    monkeypatch.setattr(
        component_specs,
        "generate_video_backbone_component_specs",
        lambda _: {
            "components": [{"attr": "dit"}, {"attr": "text_encoder"}],
        },
    )
    cfg = OmegaConf.create(
        {
            "model": {"video_backbone": {"model_path": str(tmp_path)}},
            "dataloader": {"text_embedding_cache_path": "embeddings.pt"},
        }
    )
    component_specs.save_video_backbone_deploy_assets(str(tmp_path), cfg)
    assert [c.attr for c in cfg.model.video_backbone.components] == ["dit"]
    assert cfg.model.video_backbone.precomputed_text_encoder_path == str(tmp_path)


def test_cached_text_checkpoint_restores_before_external_t5(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    import openwam.model as model
    from openwam.deploy import model_loader
    from openwam.model.video_backbone.wan import loader

    arch = _ConcreteArch(cfg=None)
    arch.video_backbone = nn.Linear(2, 2)
    ckpt = tmp_path / "checkpoint_step_1.safetensors"
    arch.save_checkpoint(str(ckpt))
    saved = arch.video_backbone.weight.detach().clone()
    with torch.no_grad():
        arch.video_backbone.weight.zero_()
    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "components": [],
                    "precomputed_text_encoder_path": "wan-weights",
                }
            },
            "training": {"mixed_precision": "no"},
        }
    )
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    monkeypatch.setattr(
        model,
        "resolve_architecture_config",
        lambda _: SimpleNamespace(
            params={},
            registry_name="tiny",
            canonical=SimpleNamespace(framework="dual_system", variant="tiny"),
        ),
    )
    monkeypatch.setattr(model, "build_architecture", lambda *args: arch)
    monkeypatch.setattr(model_loader, "_build_normalizer", lambda *args: None)
    monkeypatch.setattr(arch, "attach_normalizer", lambda *args: None)
    monkeypatch.setattr(arch, "set_dtype_device", lambda *args: None)
    t5 = nn.Linear(2, 2)

    def load_t5(path):
        assert path == "wan-weights"
        assert torch.equal(arch.video_backbone.weight, saved)
        assert not hasattr(arch.video_backbone, "text_encoder")
        return SimpleNamespace(text_encoder=t5, tokenizer="tokenizer")

    monkeypatch.setattr(loader, "load_text_components", load_t5)
    _, restored = model_loader.load_from_checkpoint_dir(str(tmp_path), device="cpu")
    assert restored.video_backbone.text_encoder is t5
    assert restored.video_backbone._tokenizer == "tokenizer"
