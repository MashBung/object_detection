import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import Conv, C2f, SPPF

class Neck(nn.Module):
    def __init__(self, channels=(128,256,512), n=1):
        super().__init__()
        ch3, ch4, ch5 = channels

        self.sppf = SPPF(ch5, ch5, k=5)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # P5 -> P4 융합: upsample된 c5(512) + c4(256) = 768 -> 256
        self.reduce_p4 = C2f(ch5+ch4, ch4,n=n,shortcut=False)

        # P4 -> P3 융합: upsample된 256 + c3(128) = 384 -> 128
        self.reduce_p3 = C2f(ch4+ch3, ch3,n=n,shortcut=False)

        self.down_p3 = Conv(ch3, ch3, kernel_size=3, stride=2)  # 40x40 -> 20x20
        self.out_p4 = C2f(ch3 + ch4, ch4, n=n, shortcut=False)  # 384 -> 256
        self.down_p4 = Conv(ch4, ch4, kernel_size=3, stride=2)  # 20x20 -> 10x10
        self.out_p5 = C2f(ch4 + ch5, ch5, n=n, shortcut=False)  # 768 -> 512

    def forward(self, c3, c4, c5):
        # c5.shape (B,512,10,10)
        p5 = self.sppf(c5) # (B,512,10,10)

        x = F.interpolate(p5, size=c4.shape[-2:], mode="nearest") # (B,512,20,20)
        p4_td = self.reduce_p4(torch.cat([x,c4],1)) # (B,768,20,20) -> (B,256,20,20)


        x = F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest") # (B,256,40,40)
        p3_out = self.reduce_p3(torch.cat([x,c3],1)) # (B,384,40,40) -> (B,128,40,40)

        # bottom-up: 위치 정보를 다시 위로
        x = self.down_p3(p3_out) # (B,128,20,20)
        p4_out = self.out_p4(torch.cat([x, p4_td],1)) # (B,256,20,20)

        x = self.down_p4(p4_out) # (B,256,10,10)
        p5_out = self.out_p5(torch.cat([x, p5], 1)) # (B,512,10,10)

        return p3_out, p4_out, p5_out
        # p3_out: (B,128,40,40)
        # p4_out: (B,256,20,20)
        # p3_out: (B,512,10,10)

