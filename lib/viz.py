import torch
from torchdiffeq import odeint_adjoint as odeint
from torchvision.utils import make_grid
from .utils import make_image
import torch.nn.functional as F


def visualize(
    fixed_w,
    generator,
    hi=2,
    steps=10,
    ode_model=None,
    n=None,
    resize=None,
    plus=False,
    n_latent=12,
    alphas=None,
    **generator_kwargs
):
    N = fixed_w.shape[0]

    imgs = []

    for i in range(N):
        w = fixed_w[i : i + 1, :]

        if alphas is not None:
            pos = [torch.linspace(0, hi, steps).to(w.device) for hi in alphas]
        else:
            pos = [torch.linspace(0, hi, steps).to(w.device) for _ in range(N)]

        with torch.no_grad():

            if plus:
                w = w.view(1, 1, -1).repeat(1, n_latent, 1).view(1, -1)

            if ode_model is not None:
                w_shifted_pos = (
                    odeint(
                        ode_model.odeblock.odefunc,
                        w,
                        pos[i],
                        rtol=1e-4,
                        atol=1e-4,
                        method="rk4",
                    )
                    .view(len(pos[i]), -1, 512)
                    .squeeze(1)
                )

            if n is not None:
                w_shifted_pos = w + n * pos[i].view(steps, 1).to(w.device)

            imgs_shifted_pos = generator([w_shifted_pos], **generator_kwargs).cpu()
            if resize:
                imgs_shifted_pos = F.interpolate(imgs_shifted_pos, size=resize)

        imgs_all = imgs_shifted_pos
        imgs.append(imgs_all)

    imgs = torch.stack(imgs, 0)
    H = imgs.shape[-1]

    grid = make_grid(imgs.reshape(-1, 3, H, H), nrow=steps)
    return make_image(grid[None, ...])[0]
