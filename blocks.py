import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = Conv(in_channels, hidden, kernel_size=1)
        self.conv2 = Conv(hidden * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super().__init__()
        hidden = int(out_channels * e)
        self.conv1 = Conv(in_channels, hidden, kernel_size=3)
        self.conv2 = Conv(hidden, out_channels, kernel_size=3)
        self.add = shortcut and (in_channels == out_channels)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(out_channels * e)
        self.conv1 = Conv(in_channels, 2 * self.c, kernel_size=1)
        self.conv2 = Conv((2 + n) * self.c, out_channels, kernel_size=1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.conv2(torch.cat(y, dim=1))


class DFLDecoder(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("dfl", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).softmax(dim=2)
        return (x * self.dfl.view(1, 1, -1, 1)).sum(dim=2)  # (32, 4, 2100)
