import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SPPF, C2f, Conv


class Neck(nn.Module):
    def __init__(self, channels=(128, 256, 512), n=1):
        super().__init__()
        ch3, ch4, ch5 = channels

        self.sppf = SPPF(ch5, ch5, k=5)

        self.reduce_p4 = C2f(ch5 + ch4, ch4, n=n, shortcut=False)

        self.reduce_p3 = C2f(ch4 + ch3, ch3, n=n, shortcut=False)

        self.up_p3 = Conv(ch3, ch3, kernel_size=3, stride=2)  # 40x40 -> 20x20
        self.out_p4 = C2f(ch3 + ch4, ch4, n=n, shortcut=False)  # 384 -> 256
        self.up_p4 = Conv(ch4, ch4, kernel_size=3, stride=2)  # 20x20 -> 10x10
        self.out_p5 = C2f(ch4 + ch5, ch5, n=n, shortcut=False)  # 768 -> 512

    def forward(self, c3, c4, c5):
        p5 = self.sppf(c5)  # (32,512,10,10)

        x = F.interpolate(p5, size=c4.shape[-2:], mode="nearest")  # (32,512,20,20)
        p4_temp = self.reduce_p4(
            torch.cat([x, c4], 1)
        )  # (32,768,20,20) > (32,256,20,20)

        x = F.interpolate(p4_temp, size=c3.shape[-2:], mode="nearest")  # (32,256,40,40)
        p3_out = self.reduce_p3(
            torch.cat([x, c3], 1)
        )  # (32,384,40,40) > (32,128,40,40)

        x = self.up_p3(p3_out)  # (32,128,20,20)
        p4_out = self.out_p4(torch.cat([x, p4_temp], 1))  # (32,256,20,20)

        x = self.up_p4(p4_out)  # (32,256,10,10)
        p5_out = self.out_p5(torch.cat([x, p5], 1))  # (32,512,10,10)

        return p3_out, p4_out, p5_out
        """
        p3_out (32,128,40,40)
        p4_out (32,256,20,20)
        p5_out (32,512,10,10)
        """
