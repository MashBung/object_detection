import torch
import torch.nn as nn
from backbone import cnn
from neck import Neck
from head import Detect

class ObjectDetection(nn.Module):
    def __init__(self, num_classes=10, channels=(128,256,512), n=1, reg_max=16):
        super().__init__()
        self.backbone = cnn()
        self.neck = Neck(channels=channels, n=n)
        self.head = Detect(num_classes=num_classes, channels=channels, reg_max=reg_max)
        self.head.bias_init()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head([p3, p4, p5])

    def load_backbone(self, path, device="cpu"):
        """가중치 백본 로드"""
        checkpoint = torch.load(path, map_location=device)
        state = {k:v for k, v in checkpoint.items() if not k.startswith("classifier")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        return missing, unexpected

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ObjectDetection(num_classes=10).to(device)

    missing, unexpected = model.load_backbone(
        r".\pretrained_model\pretrained_cnn_28.pth", device
    )
    print("missing  :", len(missing))
    print("unexpected:", len(unexpected))

    x = torch.randn(2,3,320,320).to(device)

    model.train()
    out = model(x)
    print("[train] shape:", [o.shape for o in out])

    model.eval()
    with torch.no_grad():
        y = model(x)
    print("[eval] out:", y.shape)

    total = sum(p.numel() for p in model.parameters())
    print(f"파라미터: {total/1000000:.2f}M")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습대상: {trainable/1000000:.2f}M")

    model.train()
    loss = sum(o.sum() for o in model(x))
    loss.backward()
    first = model.backbone.stem[0].weight.grad
    print("백본까지 gradient 도달:", first is not None and first.abs().sum().item() > 0)