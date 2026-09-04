import torch
import torch.nn as nn

class Conv(nn.Module):
    """Conv2d + BN + SiLU — 모든 블록의 최소 단위"""
    """Conv2d + BN + SiLU — YOLOv8 전체에서 재사용하는 기본 블록"""
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

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super().__init__()
        hidden = int(out_channels*e)
        self.conv1 = Conv(in_channels, hidden,kernel_size=3)
        self.conv2 = Conv(hidden, out_channels, kernel_size=3)
        # shortcut은 in==out일 때만 의미 있음 (더할 수 있어야 하니까)
        self.add = shortcut and (in_channels == out_channels)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = Conv(in_channels, hidden, kernel_size=1)
        self.conv2 = Conv(hidden*4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding= k //2)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))
        # (B, CH, H, W)

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(out_channels*e)
        self.conv1 = Conv(in_channels, 2 * self.c, kernel_size=1)
        # concat되는 장 수: 그대로 둔 1장 + bottleneck 통과 전 1장 + 중간 출력 n장
        self.conv2 = Conv((2 + n) * self.c, out_channels, kernel_size=1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n)
            )

    def forward(self, x):
            y = list(self.conv1(x).chunk(2, dim=1)) # 채널을 반으로 쪼갬 -> [c, c]
            for m in self.m:
                y.append(m(y[-1]))  # 직전 출력을 넣고, 결과를 목록에 누적
            return self.conv2(torch.cat(y, dim=1))

class DFL(nn.Module):
    """16-bin 분포 -> 거리 하나 (기댓값 적분)"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # 후보값 0,1,2,...,15 — 학습되지 않는 상수
        self.register_buffer("project", torch.arange(reg_max, dtype=torch.float))

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a) # 변(4개)과 bin(16개)을 분리
        x = x.softmax(dim=2)
        return (x * self.project.view(1,1,-1,1)).sum(dim=2) # (B, 4, A)

