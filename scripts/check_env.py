"""Check native training/deployment dependencies; --gpu also checks the job GPU."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Require CUDA and run forward/backward on every visible GPU")
    args = parser.parse_args()
    print(f"Python: {sys.executable}", flush=True)
    for module in (
        "torch", "torchvision", "torchaudio", "triton", "numpy", "transformers",
        "diffusers", "accelerate", "deepspeed", "hydra", "cv2", "av", "h5py",
        "pyarrow", "websockets", "msgpack", "openwam.train.openwam_trainer",
        "openwam.model.video_backbone.wan.loader", "openwam.deploy.server",
    ):
        imported = importlib.import_module(module)
        print(f"OK {module}: {getattr(imported, '__version__', '')}", flush=True)

    import torch
    from torch.utils.cpp_extension import get_compiler_abi_compatibility_and_version

    print(f"PyTorch CUDA: {torch.version.cuda}; compiler: {get_compiler_abi_compatibility_and_version('g++')}")
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the training node's device allocation and driver")
        for index in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{index}")
            x = torch.randn(128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
            loss = (x @ x.T).float().square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
                raise RuntimeError(f"Non-finite forward/backward result on {device}")
            print(f"OK {device}: {torch.cuda.get_device_name(index)}; bf16 forward/backward")
    else:
        print("GPU execution not checked; run with --gpu on the training node.")


if __name__ == "__main__":
    main()
