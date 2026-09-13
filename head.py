import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from blocks import Conv, DFLDecoder
from util import make_anchors, dist2bbox


class Detect(nn.Module):
    def __init__(self, num_classes=10, channels=(128, 256, 512), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.stride = torch.tensor([8.0, 16.0, 32.0])

        c_reg = max(16, channels[0] // 4, reg_max * 4)  # 64
        c_cls = max(channels[0], min([num_classes, 100]))  # 128

        self.reg_branch = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c_reg, kernel_size=3),
                Conv(c_reg, c_reg, kernel_size=3),
                nn.Conv2d(c_reg, 4 * reg_max, kernel_size=1),
            )
            for ch in channels
        )  # (32,64,40,40), (32,64,20,20), (32,64,10,10)

        self.cls_branch = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c_cls, kernel_size=3),
                Conv(c_cls, c_cls, kernel_size=3),
                nn.Conv2d(c_cls, num_classes, kernel_size=1),
            )
            for ch in channels
        )  # (32,10,40,40),(32,10,20,20)(32,10,10,10)

        self.anchors = None
        self.shape = None
        self.number_output = num_classes + 4 * reg_max
        self.dfl_decoder = DFLDecoder(reg_max)

    def bias_init(self):
        for reg, cls, s in zip(self.reg_branch, self.cls_branch, self.stride):
            reg[-1].bias.data[: 4 * self.reg_max] = 1.0
            cls[-1].bias.data[: self.num_classes] = math.log(
                5 / self.num_classes / (320 / s) ** 2
            )

    def forward(self, feats):
        """
        p3_out (32,128,40,40)
        p4_out (32,256,20,20)
        p5_out (32,512,10,10)
        """
        x = []
        """
        [(32,74,40,40),(32,74,20,20),(32,74,10,10)]
        """

        for i, f in enumerate(feats):
            x.append(torch.cat([self.reg_branch[i](f), self.cls_branch[i](f)], dim=1))

        if self.training:
            return x

        # ---- 추론: 하나로 합쳐 디코딩 ----
        b = x[0].shape[0]  # 32
        shape = [
            f.shape[-2:] for f in feats
        ]  # [Size([40,40]), Size([20,20]), Size([10,10])]

        if self.shape != shape:
            self.anchors, self.strides = make_anchors(feats, self.stride)
            self.anchors = self.anchors.transpose(0, 1)  # (2,2100)
            self.strides = self.strides.transpose(0, 1)  # (1,2100)
            self.shape = shape

        x_cat = torch.cat([idx_x.view(b, self.number_output, -1) for idx_x in x], dim=2)
        # (32,76,2100)
        box, cls = x_cat.split((4 * self.reg_max, self.num_classes), dim=1)
        # 64, 10

        dist = self.dfl_decoder(box)  # (32,4,2100)
        dbox = dist2bbox(dist, self.anchors.unsqueeze(0), dim=1) * self.strides

        return torch.cat([dbox, cls.sigmoid()], dim=1)  # (32,14,2100)


if __name__ == "__main__":
    p3 = torch.randn(1, 128, 40, 40)
    p4 = torch.randn(1, 256, 20, 20)
    p5 = torch.randn(1, 512, 10, 10)
    feats = [p3, p4, p5]

    head = Detect(num_classes=10, channels=(128, 256, 512))
    head.bias_init()
    head.train()

    outs = head(feats)
    print("[train] 반환 개수:", len(outs))
    print("[train] shapes  :", [o.shape for o in outs])

    head.eval()
    with torch.no_grad():
        y = head(feats)
    print("[eval] out:", y.shape)

    boxes, scores = y[:, :4], y[:, 4:]
    print("box 범위 :", boxes.min().item(), boxes.max().item())
    print("score 범위:", scores.min().item(), scores.max().item())
    print("유효 박스:", bool((boxes[:, 0] <= boxes[:, 2]).all()))
