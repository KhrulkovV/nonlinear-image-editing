import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from lib.constants import CARDINALITY
from scipy.stats import entropy
from scipy.special import entr
from tqdm import tqdm
import torchvision.models as tv
import os


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False, size=64
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        second_stride = 2 if size > 32 else 1
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=second_stride)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class CNN_Encoder(nn.Module):
    def __init__(self, in_channel=3, size=64):
        super().__init__()
        init_stride = 2 if size == 64 else 1
        init_padding = 1 if size == 64 else 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, init_stride, init_padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.flatten(self.encoder(x), 1)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def cnn_encoder(**kwargs):
    return CNN_Encoder(**kwargs)


class CelebaRegressor(nn.Module):
    def __init__(
        self,
        discrete_cardinality,
        cont_cardinality,
        f_size=512,
        pretrained=False,
        **bb_kwargs,
    ):
        super().__init__()
        encoder = tv.resnet18(pretrained=pretrained)
        encoder.fc = nn.Sequential(nn.ReLU(), nn.Linear(512, f_size), nn.ReLU())
        self.backbone = encoder

        self.classification_heads = nn.ModuleList(
            [nn.Linear(f_size, d) for d in discrete_cardinality]
        )
        self.continuous_head = nn.Linear(f_size, cont_cardinality)  # 10

    def forward(self, x):
        features = self.backbone(x)
        d_features = [head(features) for head in self.classification_heads]
        c_features = self.continuous_head(features)
        return d_features


class FactorRegressor(nn.Module):
    def __init__(
        self,
        discrete_cardinality,
        backbone="resnet18",
        f_size=256,
        pretrained=False,
        **bb_kwargs,
    ):
        super().__init__()
        features = 512 if "resnet" in backbone else 1024

        if pretrained:
            encoder = tv.resnet18(pretrained=True)
            for p in encoder.parameters():
                p.requires_grad_(False)

            encoder.fc = nn.Sequential(
                nn.ReLU(), nn.Linear(features, f_size), nn.ReLU(),
            )
            self.backbone = encoder
        else:
            self.backbone = nn.Sequential(
                globals()[backbone](**bb_kwargs),
                nn.ReLU(),
                nn.Linear(features, f_size),
                nn.ReLU(),
            )

        self.classification_heads = nn.ModuleList(
            [nn.Linear(f_size, d) for d in discrete_cardinality]
        )

    def forward(self, x):
        features = self.backbone(x)
        discr = [head(features) for head in self.classification_heads]
        return discr


class ImagenetNormalize(nn.Module):
    """
    Converts [-1, 1] image normalization to the ImageNet one
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std


class Scaler(nn.Module):
    def forward(self, x):
        return (x + 1) / 2


class FullCrossEntropy(nn.Module):
    def forward(self, x, y):
        return F.kl_div(F.log_softmax(x, dim=1), y, reduction="batchmean")


class Constant(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        return x * 0 + self.const


class ResLinear(nn.Linear):
    def forward(self, x):
        out = super().forward(x)
        # assert out.shape[1] == x.shape[1]
        return out + x


class ODEfunc(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        if depth == -1:
            self.model = Constant(dim)
        else:
            layers = [
                ResLinear(dim, dim),
            ]
            for i in range(depth - 1):
                layers.extend([nn.LeakyReLU(0.2), ResLinear(dim, dim)])
            self.model = nn.Sequential(*layers)
        self.depth = depth
        self.nfe = 0

    def forward(self, t, x, eps=1e-6):
        self.nfe += 1
        out = self.model(x)
        out = out / (torch.norm(out, dim=1, keepdim=True) + eps)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, x, alpha=3, sign=1):
        integration_time = torch.tensor([0, alpha]).float()
        self.integration_time = integration_time.type_as(x).to(x.device)
        self.odefunc.sign = sign
        out = odeint(
            self.odefunc, x, self.integration_time, rtol=1e-4, atol=1e-4, method="rk4"
        )
        return out[1]

    def forward_with_time(self, x, alpha=3, steps=8):
        integration_time = torch.linspace(0, alpha, steps).float().to(x.device)
        self.odefunc.sign = 1
        out = odeint(
            self.odefunc, x, integration_time, rtol=1e-4, atol=1e-4, method="rk4"
        )
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ResizeTo(nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size)


def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


class OdeRectifier(nn.Module):
    def __init__(
        self,
        dataset,
        generator,
        regressor,
        device,
        postprocessing_net=None,
        depth=1,
        idx=0,
        **generator_kwargs,
    ):
        super().__init__()

        self.generator = generator
        self.regressor = regressor
        self.generator_kwargs = generator_kwargs
        self.cardinality = CARDINALITY[dataset]

        self.dim = self.generator.style_dim
        self.odefunc = ODEfunc(self.dim, depth=depth)
        self.odeblock = ODEBlock(self.odefunc)
        self.postprocessing_net = postprocessing_net
        print("ODE:", self.odefunc)
        data = torch.load(f"data_to_rectify/{dataset}_all.pt")
        self.data = data["ws"][np.where(data["labels"][:, idx] == 0)]

        self.device = device
        self.generator.eval()
        self.regressor.eval()

        for p in self.generator.parameters():
            p.requires_grad_(False)

        for p in self.regressor.parameters():
            p.requires_grad_(False)

    def forward(self, batch_size=64, alpha=12):
        # z = torch.randn(batch_size, self.generator.style_dim).to(self.device)
        idx = np.random.choice(np.arange(len(self.data)), batch_size)
        w = self.data[idx]
        w = torch.from_numpy(w).float().to(self.device)
        value = np.random.uniform(alpha / 4, alpha)
        with torch.no_grad():
            # w = self.generator.style(z)

            predicts_orig = self.regressor(
                self.postprocessing_net(
                    torch.clamp(self.generator([w], **self.generator_kwargs), -1, 1)
                )
            )

        shifted = self.odeblock(w, value, sign=1)

        predicts_shifted = self.regressor(
            self.postprocessing_net(
                torch.clamp(self.generator([shifted], **self.generator_kwargs), -1, 1,)
            )
        )
        return (w, shifted), (predicts_orig, predicts_shifted)

    @torch.no_grad()
    def validate(self, size=1000, idx=2, alpha=8, steps=40, batch_size=100):
        print("Evaluating.")
        labels = []
        t = np.linspace(0, alpha, steps)
        for t_ in t:
            labels_ = []
            for i in tqdm(range(size // batch_size)):
                w_ = self.data[i * batch_size : (i + 1) * batch_size]
                w_ = torch.from_numpy(w_).float().to(self.device)
                if t_ > 0:
                    shifted = self.odeblock(w_, t_, sign=1)
                else:
                    shifted = w_

                predicts_shifted = self.regressor(
                    self.postprocessing_net(
                        torch.clamp(
                            self.generator([shifted], **self.generator_kwargs), -1, 1,
                        )
                    )
                )
                label = torch.stack([d.argmax(1) for d in predicts_shifted], 1)

                labels_.append(label.cpu().numpy())
            labels_ = np.concatenate(labels_, 0)

            labels.append(labels_)

        lbl = np.stack(labels, -1)

        change = []
        for j in range(lbl.shape[2]):
            diff = np.any(lbl[:, idx, : (j + 1)] != lbl[:, idx, 0:1], axis=1)
            change.append(np.mean(diff))

        disentangle = []
        for j in range(lbl.shape[2]):
            dis_ = []
            for i in set(np.arange(lbl.shape[1])) - {idx}:
                counts = bincount2d(lbl[:, i, : (j + 1)], bins=self.cardinality[i])
                proba = counts / np.sum(counts, 1, keepdims=True)
                dis = np.mean(
                    np.sum(entr(proba), 1) / entropy(np.ones(self.cardinality[i]))
                )
                dis_.append(dis)
            disentangle.append(np.mean(dis_))

        return change, disentangle
