"""Precompute Wan T5 embeddings from a LIBERO LeRobot v3 data config."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from openwam.model.video_backbone.wan.encode import encode_text, text_cache_metadata
from openwam.model.video_backbone.wan.loader import load_text_components


def collect_prompts(data_config) -> list[str]:
    if data_config.type != "libero":
        raise ValueError("T5 precompute currently supports LIBERO LeRobot v3 data configs")
    root = Path(data_config.dataset_dir)
    buckets = (
        [root]
        if (root / "meta/info.json").is_file()
        else sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta/info.json").is_file())
    )
    if not buckets:
        raise FileNotFoundError(f"No LeRobot v3 datasets under {root}")
    prompts = {""}
    for bucket in buckets:
        tasks = pd.read_parquet(bucket / "meta/tasks.parquet")
        for text in tasks.index:
            prompt = text.strip()
            if not prompt:
                raise ValueError(f"Blank task text in {bucket}/meta/tasks.parquet")
            prompts.add(prompt)
    return sorted(prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-path", default="/mnt/cpfs/feiyang/checkpoints/Wan2.2-TI2V-5B")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.data_config)
    if not cfg.get("text_embedding_cache_path"):
        raise ValueError("Set text_embedding_cache_path in the data config")
    prompts = collect_prompts(cfg)
    output = Path(cfg.text_embedding_cache_path)
    holder = load_text_components(args.model_path, device="cuda")
    embeddings = {}
    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            context, seq_lens = encode_text(
                [prompt], tokenizer=holder.tokenizer, text_encoder=holder.text_encoder, device="cuda"
            )
            embeddings[prompt] = (context.cpu(), seq_lens.cpu())
            print(f"Encoded {i}/{len(prompts)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as f:
            temporary = f.name
            torch.save({"metadata": text_cache_metadata(args.model_path), "embeddings": embeddings}, f)
        os.replace(temporary, output)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Saved {len(embeddings)} prompts to {output}")


if __name__ == "__main__":
    main()
