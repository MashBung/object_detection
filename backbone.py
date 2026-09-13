import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        if stride != 1 or in_channels != out_channels:
            self.pipe = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.pipe = nn.Identity()

    def forward(self, x):
        pipe_x = self.pipe(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(pipe_x + out)


class cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )  # (32,32,160,160)

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )  # (32,32,160,160)

        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )  # (32,64,80,80)

        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )  # (32,128,40,40)

        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )  # (32,256,20,20)

        self.stage5 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )  # (32,512,10,10)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5
        """
        C3: torch.Size([b, 128, 40, 40])
        C4: torch.Size([b, 256, 20, 20])
        C5: torch.Size([b, 512, 10, 10])
        """


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = cnn().to(device)
    checkpoint = torch.load(
        r".\pretrained_model\pretrained_cnn_23.pth",
        map_location=device,
    )

    backbone.load_state_dict(checkpoint, strict=False)

    backbone.eval()

    x = torch.randn(1, 3, 320, 320).to(device)
    c3, c4, c5 = backbone(x)

    print("C3:", c3.shape)
    print("C4:", c4.shape)
    print("C5:", c5.shape)
