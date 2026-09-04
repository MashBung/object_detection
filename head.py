import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from blocks import DFL, Conv
from utils import make_anchors, dist2bbox

class Detect(nn.Module):
    def __init__(self, num_classes=10, channels=(128,256,512), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.number_level = len(channels)
        self.number_output = num_classes + 4 * reg_max
        self.stride = torch.tensor([8.0,16.0,32.0])

        c_reg = max(16, channels[0]//4, reg_max*4)
        c_cls = max(channels[0], min(num_classes,100))

        self.reg_branch = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c_reg, kernel_size=3),
                Conv(c_reg, c_reg, kernel_size=3),
                nn.Conv2d(c_reg, 4*reg_max, kernel_size=1),
            )
            for ch in channels
        )
        self.cls_branch = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c_cls, kernel_size=3),
                Conv(c_cls, c_cls, kernel_size=3),
                nn.Conv2d(c_cls, num_classes, kernel_size=1),
            )
            for ch in channels
        )

        self.dfl = DFL(reg_max)
        self.anchors = None
        self.shape = None

    def bias_init(self):
        for reg, cls, s in zip(self.reg_branch, self.cls_branch, self.stride):
            reg[-1].bias.data[:4 * self.reg_max] = 1.0
            # 이미지당 객체 n개, 320*320 기준 사전확률
            cls[-1].bias.data[: self.num_classes] = math.log(5 / self.num_classes / (320 / s) ** 2)

    def forward(self, feats):
        x = []
        for i, f in enumerate(feats):
            x.append(torch.cat([self.reg_branch[i](f), self.cls_branch[i](f)], dim=1))

        if self.training:
            return x
            # x (B, 74, 40, 40), (B, 74, 20, 20), (B, 74, 10, 10)

        # ---- 추론: 하나로 합쳐 디코딩 ----
        b = x[0].shape[0]
        shape = [f.shape[-2:] for f in feats]

        # 입력 해상도가 바뀔 때만 anchor 재생성
        if self.shape != shape:
            self.anchors, self.strides = make_anchors(feats, self.stride)
            self.anchors = self.anchors.transpose(0, 1)
            self.strides = self.strides.transpose(0, 1)
            self.shape = shape

        x_cat = torch.cat([xi.view(b, self.number_output,-1) for xi in x], dim=2)
        box, cls = x_cat.split((4 * self.reg_max, self.num_classes), dim=1)

        dist = self.dfl(box) # (B, 4, A)
        dbox = dist2bbox(dist, self.anchors.unsqueeze(0), dim=1) * self.strides

        return torch.cat([dbox, cls.sigmoid()], dim=1) # (B, 4+nc, A)
            

if __name__ == "__main__":
    p3 = torch.randn(1, 128, 40, 40)
    p4 = torch.randn(1, 256, 20, 20)
    p5 = torch.randn(1, 512, 10, 10)
    feats = [p3, p4, p5]

    head = Detect(num_classes=10, channels=(128,256,512))
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


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# from blocks import Conv, DFL
# from utils import make_anchors, dist2bbox

# class Detect(nn.Module):
#     def __init__(self, num_classes=10, channels=(128,256,512), reg_max=16):
#         super().__init__()
#         self.num_classes = num_classes
#         self.reg_max = reg_max
#         self.number_level = len(channels)
#         self.number_output = num_classes + 4 * reg_max
#         self.stride = torch.tensor([8.0,16.0,32.0])

#         c_reg = max(16, channels[0] // 4, reg_max * 4)
#         c_cls = max(channels[0], min(num_classes, 100))

#         self.reg_branch = nn.ModuleList(
#             nn.Sequential(
#                 Conv(ch, c_reg, kernel_size=3),
#                 Conv(c_reg, c_reg, kernel_size=3),
#                 nn.Conv2d(c_reg, 4 * reg_max, kernel_size=1),
#             )
#             for ch in channels
#         )
#         self.cls_branch = nn.ModuleList(
#             nn.Sequential(
#                 Conv(ch, c_cls, kernel_size=3),
#                 Conv(c_cls, c_cls, kernel_size=3),
#                 nn.Conv2d(c_cls, num_classes, kernel_size=1),
#             )
#             for ch in channels
#         )

#         self.dfl = DFL(reg_max)
#         self.anchors = None
#         self.shape = None

#     def bias_init(self):
#         for reg, cls, s in zip(self.reg_branch, self.cls_branch, self.stride):
#             reg[-1].bias.data[:] = 1.0
#             cls[-1].bias.data[: self.num_classes] = math.log(5 / self.num_classes / (640 / s) ** 2)

#     def forward(self, feats):
#         x = []
#         for i, f in enumerate(feats):
#             x.append(torch.cat([self.reg_branch[i](f), self.cls_branch[i](f)], dim=1))

#         if self.training:
#             return x

#         # ---- 추론: 하나로 합쳐 디코딩 ----
#         b = x[0].shape[0]
#         shape = [f.shape[-2:] for f in feats]

#         # ---- 추론: 하나로 합쳐 디코딩 ----
#         b = x[0].shape[0]
#         shape = [f.shape[-2:] for f in feats]

#         # 입력 해상도가 바뀔 때만 anchor 재생성
#         if self.shape != shape:
#             self.anchors, self.strides = make_anchors(feats, self.stride)
#             self.anchors = self.anchors.transpose(0, 1)  # (2, A)
#             self.strides = self.strides.transpose(0, 1)  # (1, A)
#             self.shape = shape

#         # (B, no, H, W) 3장 -> (B, no, A)
#         x_cat = torch.cat([xi.view(b, self.number_output, -1) for xi in x], dim=2)
#         box, cls = x_cat.split((4 * self.reg_max, self.num_classes), dim=1)

#         dist = self.dfl(box)  # (B, 4, A)
#         dbox = dist2bbox(dist, self.anchors.unsqueeze(0), dim=1) * self.strides

#         return torch.cat([dbox, cls.sigmoid()], dim=1)  # (B, 4+nc, A)

# if __name__ == "__main__":
#     p3 = torch.randn(32,128,40,40)
#     p4 = torch.randn(32,256,20,20)
#     p5 = torch.randn(32,512,10,10)
#     feats = [p3,p4,p5]

#     model = Detect(num_classes=10)
#     model.train()
#     outs = model(feats)
#     print("[train] 반환 개수:", len(outs))
#     print("[train] shapes  :", [o.shape for o in outs])
    
#     model.bias_init()
#     for i, cls in enumerate(model.cls_branch):
#         print(f"level {i}: {cls[-1].bias.data[0].item():.2f}")

#     model.eval()
#     with torch.no_grad():
#         y = model(feats)
#     print("[eval] out:", y.shape)

#     boxes, scores = y[:, :4], y[:, 4:]
#     print("box 범위 :", boxes.min().item(), boxes.max().item())
#     print("score 범위:", scores.min().item(), scores.max().item())
#     print("유효 박스:", bool((boxes[:, 0] <= boxes[:, 2]).all()))