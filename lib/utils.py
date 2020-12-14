import torch
import torch.nn as nn
from .constants import GENERATOR_KWARGS, ATTRIBUTES, IDX2ATTR
import json
from torchdiffeq import odeint_adjoint as odeint
from lib.stylegan2.models import Generator
from lib.models import ODEBlock, ODEfunc
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid
from typing import Union, Tuple, List, Optional
import imageio


def make_image(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .clamp(min=-1, max=1)
        .add(1)
        .div(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


class FlowFactory:
    def __init__(
        self,
        dataset: str,
        dim: int = 512,
        root_dir: Union[str, Path] = Path("./log"),
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset = dataset
        self.root_dir = Path(root_dir)
        self.dim = dim
        self.device = device

    def _validate_idx(self, i: int) -> bool:
        return i in IDX2ATTR[self.dataset]

    def _validate_name(self, s: str) -> bool:
        return s in ATTRIBUTES[self.dataset]

    def _validate_name_or_idx(self, x: Union[int, str]) -> bool:
        if isinstance(x, int):
            if not self._validate_idx(x):
                raise ValueError(
                    f"Incorrect attribute index: "
                    f"\n{IDX2ATTR[self.dataset]} are available;"
                    f"\nGot: {x}"
                )

        elif isinstance(x, str):
            if not self._validate_name(x):
                raise ValueError(
                    f"Incorrect attribute index: "
                    f"\n{ATTRIBUTES[self.dataset]} are available;"
                    f"\nGot: {x}"
                )

        else:
            raise ValueError(f"Incorrect data type.")

    def _list_available(self) -> Tuple[List[str], List[int]]:
        return (ATTRIBUTES[self.dataset], IDX2ATTR[self.dataset])

    def _idx2name(self, i: int) -> str:
        return ATTRIBUTES[self.dataset][i]

    def _name2idx(self, s: str) -> int:
        idx = ATTRIBUTES[self.dataset].index(s)
        return IDX2ATTR[self.dataset][idx]

    def _build_odeblock(self, depth: int = 1) -> nn.Module:
        odefunc = ODEfunc(dim=self.dim, depth=depth)
        odeblock = ODEBlock(odefunc)
        return odeblock

    def _load_from_checkpoint(
        self, i: Optional[int] = None, s: Optional[str] = None, depth: int = 1
    ) -> "LatentFlow":
        if (i is not None) and (s is not None):
            raise ValueError(f"Both i and s can not be specified: got {i} and {s}.")

        odeblock = self._build_odeblock(depth=depth)

        if i is not None:
            self._validate_name_or_idx(i)
            s = self._idx2name(i)
        else:
            self._validate_name_or_idx(s)
            i = self._name2idx(s)

        checkpoint_path = self.root_dir / self.dataset / f"{depth}" / f"ckpt_{i}.pt"
        if not checkpoint_path.exists():
            raise ValueError(f"{checkpoint_path} does not exist")

        w = torch.load(checkpoint_path, map_location="cpu",)
        odeblock.load_state_dict(w)
        odeblock = odeblock.to(self.device)
        return LatentFlow(odeblock.odefunc, device=self.device, name=s)

    def _load_from_vector(
        self, n: np.ndarray, name: Optional[str] = None
    ) -> "LatentFlow":
        if len(n.shape) == 1:
            n = n.reshape(1, -1)
        assert n.shape[1] == self.dim
        n /= np.linalg.norm(n, dim=-1)
        n = torch.from_numpy(n).float()
        odeblock = self._build_odeblock(depth=0)
        odeblock.odefunc.const.data.copy_(n)
        odeblock = odeblock.to(self.device)
        return LatentFlow(odeblock.odefunc, device=self.device, name=name)


class LatentFlow:
    """
    A simple wrapper that
    computes trajectories produced by Neural ODE
    in the latent space.
    """

    def __init__(self, odefunc, device, name):
        self.odefunc = odefunc
        self.device = device
        self.name = name

    @torch.no_grad()
    def flow(self, w, t, truncate_real=6):
        """
        Computes the flow of Neural ODE starting at w.
        """
        if isinstance(t, float):
            t = torch.FloatTensor([0, t]).to(self.device)

        if not isinstance(t, torch.Tensor):
            t = torch.FloatTensor(t).to(self.device)

        if not isinstance(w, torch.Tensor):
            w = torch.FloatTensor(w).to(self.device)

        if len(w.shape) == 1:
            w = w.view(1, -1)

        # TO ACCOUNT FOR W-PLUS
        if len(w.shape) == 3:
            w_to_flow = w[:, :truncate_real, :]
            w_keep = w[:, truncate_real:, :]
            out_shape = [len(t), *w_to_flow.shape]
            out = odeint(
                self.odefunc,
                w_to_flow.reshape(-1, out_shape[-1]),
                t,
                rtol=1e-4,
                atol=1e-4,
                method="rk4",
            ).view(*out_shape)
            # T x B x L x D
            return torch.cat((out, w_keep.repeat(len(t), 1, 1, 1)), dim=2)

        else:
            out = odeint(self.odefunc, w, t, rtol=1e-4, atol=1e-4, method="rk4",)
            return out

    def __str__(self):
        if self.name is not None:
            s = (
                f"Latent Flow for the attribute {self.name} "
                f"with RHS of depth {self.odefunc.depth}."
            )
        return s


class LatentSpaceExplorer:
    def __init__(
        self,
        dataset: str,
        device: torch.device,
        data_root: Union[str, Path] = Path("."),
    ):
        config_path = data_root / "configs" / f"{dataset}.json"
        with open(config_path, "r") as f:
            generator_config = json.load(f)

        print("Building generator.")
        generator = Generator(**generator_config)

        checkpoint_path = data_root / "models" / f"{dataset}" / "generator.pth"
        ckpt = torch.load(checkpoint_path)["g_ema"]
        generator.load_state_dict(ckpt)

        self.generator = generator.eval().to(device)
        self.device = device
        with torch.no_grad():
            mean_latent = self.generator.mean_latent(4096)

        self.generator_kwargs = {**GENERATOR_KWARGS, "truncation_latent": mean_latent}

        self.flow_factory = FlowFactory(
            dataset=dataset,
            dim=generator.style_dim,
            root_dir=data_root / "log",
            device=device,
        )

        self.logdir = data_root / "results"
        if not self.logdir.exists():
            print(f"Creating directory for results: {self.logdir}.")
            self.logdir.mkdir()
        else:
            print(f"Will log results to: {self.logdir}.")

        self.dataset = dataset

    @torch.no_grad()
    def random_sample(self) -> Image:
        z = torch.randn(1, 512).to(self.device)
        w = self.generator.style(z)
        x = self.generator([w], **self.generator_kwargs)
        x = make_image(x)[0]
        return Image.fromarray(x)

    @torch.no_grad()
    def generate_latent(self, size: int = 4) -> torch.Tensor:
        z = torch.randn(size, self.generator.style_dim).to(self.device)
        w = self.generator.style(z)
        return w

    @torch.no_grad()
    def visualize_latent_grid(
        self, w: torch.Tensor, nrow: Optional[int] = None
    ) -> Image.Image:
        """
        @w : B x D
        Generates an image grid with nrow rows.
        """
        if nrow is None:
            nrow = int(np.sqrt(w.shape[0]))
        x = self.generator([w], **self.generator_kwargs)
        img = make_grid(x, nrow=x.shape[0] // nrow)
        img = make_image(img[None, ...])[0]
        return Image.fromarray(img)

    @torch.no_grad()
    def get_flow_frames(
        self,
        w: torch.Tensor,
        hi: float,
        steps: int,
        attr_name: Optional[str] = None,
        attr_idx: Optional[int] = None,
        depth: int = 1,
        save_gif: bool = False,
        save_as_mp4: bool = False,
        duration: float = 5.0,
        truncate_real: int = 6,
    ) -> List[Image.Image]:
        """
        @w: B x D or B x L x D
        Flow each style vector w with a Neural ODE flow specified by 'attr_name'.
        :rtype: List of images (for each time step we get an image grid)
        """
        flow = self.flow_factory._load_from_checkpoint(
            i=attr_idx, s=attr_name, depth=depth
        )
        w_out = flow.flow(w, np.linspace(0, hi, steps), truncate_real=truncate_real)
        w_out = torch.unbind(w_out, 0)
        imgs = [self.visualize_latent_grid(w) for w in w_out]

        if save_gif:
            if save_as_mp4:
                save_path = self.logdir / ("_".join([self.dataset, flow.name]) + ".mp4")
                writer = imageio.get_writer(
                    save_path,
                    mode="I",
                    fps=(2 * len(imgs) - 1) / duration,
                    codec="libx264",
                    bitrate="16M",
                )
            else:
                save_path = self.logdir / ("_".join([self.dataset, flow.name]) + ".gif")
                writer = imageio.get_writer(
                    save_path, mode="I", duration=duration / (2 * len(imgs) - 1)
                )

            for im in imgs:
                writer.append_data(np.array(im))
            for im in reversed(imgs[:-1]):
                writer.append_data(np.array(im))
            writer.close()

        return imgs

    @torch.no_grad()
    def get_flow_grid(
        self,
        w: torch.Tensor,
        hi: float,
        steps: int,
        attr_name: Optional[str] = None,
        attr_idx: Optional[int] = None,
        depth: int = 1,
        truncate_real: int = 6,
    ) -> Image.Image:
        """
        @w: [B x D] or [B x L x D]
        Flow each style vector w with a Neural ODE flow specified by 'attr_name'.
        :rtype: generates an image grid of shape len(w) x steps.
        """
        flow = self.flow_factory._load_from_checkpoint(
            i=attr_idx, s=attr_name, depth=depth
        )
        w_out = flow.flow(w, np.linspace(0, hi, steps), truncate_real=truncate_real)
        if len(w_out.shape) == 4:
            shape = w_out.shape
            B = w_out.shape[1]
            w_out = w_out.transpose(0, 1).reshape(-1, shape[2], shape[3])
        else:
            shape = w_out.shape
            B = w_out.shape[1]
            w_out = w_out.transpose(0, 1).reshape(-1, shape[2])

        img = self.visualize_latent_grid(w_out, nrow=B)
        return img
