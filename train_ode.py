from lib.models import (
    FactorRegressor,
    FullCrossEntropy,
    OdeRectifier,
    CelebaRegressor,
    ImagenetNormalize,
    ResizeTo,
    Scaler,
)
import argparse
import os
import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import json
from lib.constants import CARDINALITY, BB_KWARGS
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from lib.viz import visualize
from PIL import Image
from lib.stylegan2.models import Generator


try:
    import wandb

    wandb.init()
    USE_WANDB = True
except:
    print("wandb not available.")
    USE_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Neural ODE model for semantic image manipulation"
    )
    parser.add_argument(
        "--dataset",
        default="mpi3d",
        choices=["mpi3d", "ffhq", "isaac3d", "cub", "scenes"],
    )
    parser.add_argument("--nb-iter", default=1000, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--dir", default=None, nargs="+", type=int)
    parser.add_argument("--prefix", default="./log", type=str)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=6)
    parser.add_argument("--checkpoint-root", default="./models", type=str)
    parser.add_argument("--validate", default=False, action="store_true")
    parser.add_argument("--temperature", default=4.0, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    prefix = os.path.join(args.prefix, args.dataset, f"{args.depth}")
    os.makedirs(prefix, exist_ok=True)

    if USE_WANDB:
        wandb.config.depth = args.depth
        wandb.dir = args.dir
        wandb.run.name = wandb.run.id

    dataset = args.dataset

    cardinality = CARDINALITY[dataset]
    bb_kwargs = BB_KWARGS[dataset]

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    with open(f"./configs/{args.dataset}.json", "r") as f:
        generator_config = json.load(f)

    generator = Generator(**generator_config)

    path = os.path.join(args.checkpoint_root, args.dataset, "generator.pth")
    ckpt = torch.load(path)["g_ema"]
    generator.load_state_dict(ckpt)
    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        trunc = generator.mean_latent(4096)

    if dataset == "isaac3d":
        truncation = 1.0
    else:
        truncation = 0.7

    generator_kwargs = {
        "input_is_latent": True,
        "randomize_noise": False,
        "truncation_latent": trunc,
        "truncation": truncation,
    }

    if args.dataset == "ffhq":
        regressor = CelebaRegressor(
            discrete_cardinality=[2] * 40, cont_cardinality=10, f_size=512
        )
    else:
        if args.dataset in ["cub", "scenes"]:
            pretrained = True
        else:
            pretrained = False
        regressor = FactorRegressor(
            backbone="cnn_encoder" if dataset == "mpi3d" else "resnet18",
            discrete_cardinality=cardinality,
            pretrained=pretrained,
            **bb_kwargs,
        )

    path = os.path.join(args.checkpoint_root, args.dataset, "regressor.pth")
    regressor.load_state_dict(torch.load(path)["model_state_dict"])
    regressor = regressor.to(device).eval()

    criterion = FullCrossEntropy()

    if args.dataset in ["ffhq", "cub", "scenes"]:
        postprocessing_net = nn.Sequential(ResizeTo(size=224), ImagenetNormalize())
    elif args.dataset == "isaac3d":
        postprocessing_net = Scaler()
    else:
        postprocessing_net = nn.Identity()

    ws = torch.load(f"data_to_rectify/{dataset}_all.pt")["ws"]
    labels = torch.load(f"data_to_rectify/{dataset}_all.pt")["labels"]

    tau = args.temperature
    alpha = args.alpha

    def _train_single(idx):

        fixed_w = (
            torch.from_numpy(ws[np.where(labels[:, idx] == 0)]).float().to(device)[:10]
        )

        writer = SummaryWriter(f"{prefix}")

        model = OdeRectifier(
            dataset,
            generator,
            regressor,
            device,
            postprocessing_net=postprocessing_net,
            depth=args.depth,
            idx=idx,
            **generator_kwargs,
        ).to(device)
        optimizer = torch.optim.Adam(model.odeblock.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.5
        )
        remaining_idx = set(np.arange(len(cardinality))) - {idx}

        pbar = range(args.nb_iter + 1)
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.01)
        pbar.set_description(f"DIR: {idx}")

        ce = nn.CrossEntropyLoss()

        for i in pbar:
            optimizer.zero_grad()

            _, (orig, shifted) = model(batch_size=args.batch_size, alpha=alpha)

            target = (cardinality[idx] - 1) * torch.ones(args.batch_size).to(device)
            target = target.long()

            loss_main = 0
            loss_main += ce(shifted[idx] / tau, target)

            accuracy = torch.mean(
                (torch.max(shifted[idx], 1)[1] == target).float().mean()
            )

            loss_other = 0
            for j in remaining_idx:
                loss_other += criterion(
                    shifted[j] / tau, torch.softmax(orig[j].detach(), 1)
                )

            loss = loss_main + args.weight * loss_other / (len(remaining_idx))

            loss.backward()
            optimizer.step()

            writer.add_scalar(f"loss_{idx}", loss.item(), i)
            if USE_WANDB:
                wandb.log(
                    {f"loss_{idx}": loss.item(), f"accuracy_{idx}": accuracy.item()}
                )

            pbar.set_postfix(loss=loss.item(), acc=accuracy.item())

            if i % 1000 == 0:
                images = visualize(
                    fixed_w,
                    generator,
                    ode_model=model,
                    resize=None,
                    hi=alpha,
                    plus=False,
                    steps=20,
                    **generator_kwargs,
                )
                torch.save(model.odeblock.state_dict(), f"{prefix}/ckpt_{idx}.pt")
                images = Image.fromarray(images)
                images.save(f"{prefix}/example_{idx}.png")

                if args.validate:
                    change, disentangle = model.validate(
                        idx=idx, size=2000, batch_size=100, alpha=alpha, steps=40
                    )
                    torch.save((change, disentangle), f"{prefix}/curve_{idx}_{i}.pt")
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.plot(change, disentangle)
                    ax.set_xlabel("Target factor shift.")
                    ax.set_ylabel("Disentanglement.")

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    curve = Image.open(buf)
                    curve.save(f"{prefix}/curve_{idx}_{i}.png")

                if USE_WANDB:
                    wandb.log(
                        {
                            "examples": [
                                wandb.Image(images, caption=f"DIR: {idx} at iter {i}")
                            ]
                        }
                    )
                    if args.validate:
                        wandb.log(
                            {
                                "curve": [
                                    wandb.Image(curve, caption=f"Curve at iter {i}")
                                ]
                            }
                        )
            scheduler.step()

        return model

    if args.dir is None:
        print("Rectifying all the directions.")
        for i in tqdm(range(len(cardinality))):
            model = _train_single(i)
    else:
        print(f"Rectifying directions: {args.dir}.")
        for j in args.dir:
            model = _train_single(j)
