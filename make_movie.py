from lib.utils import LatentSpaceExplorer
import torch
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Neural ODE model for semantic image manipulation"
    )
    parser.add_argument(
        "--dataset", default="ffhq", choices=["ffhq", "cub", "scenes"],
    )
    parser.add_argument(
        "--w-path", default=None, type=str, help="Path to precomputed style vectors."
    )
    parser.add_argument("--nrow", default=2, type=int)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--attribute", type=str)
    parser.add_argument("--format", type=str, default="mp4", choices=["gif", "mp4"])
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hi", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    explorer = LatentSpaceExplorer(args.dataset, device=device)
    if args.w_path is not None:
        w = torch.load(args.w_path).to(device)
    else:
        w = explorer.generate_latent(args.nrow ** 2)

    explorer.get_flow_frames(
        w,
        hi=args.hi,
        steps=args.steps,
        attr_name=args.attribute,
        save_gif=True,
        save_as_mp4=(args.format == "mp4"),
        duration=args.duration,
        depth=args.depth,
    )
