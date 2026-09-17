from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_does_not_warm_start_from_a_placeholder_by_default():
    config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    assert config["training"]["finetune_ckpt_path"] is None


def test_training_output_uses_one_timestamp_free_root_key():
    config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    assert "output_path" not in config["training"]
    assert "${now" not in config["project"]["output_dir"]
    assert config["hydra"]["run"]["dir"].endswith("/${now:%Y-%m-%d_%H-%M-%S}/logs")


def test_robocoin_trim_manifest_is_opt_in():
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "dataloader" / "pretrain_data" / "robocoin.yaml").read_text(encoding="utf-8")
    )

    assert config["trim_csv"] is None
