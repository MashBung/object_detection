import torch
from torch.utils.data import DataLoader
from model import ObjectDetection
from loss import DetectionLoss
from dataset import ObjectDetectionDataset, collate_fn
from functools import partial


def train(epochs=50, device="cpu", pretrained=None):
    num_classes = 10
    img_size = 320

    dataset = ObjectDetectionDataset("./data/images/train", img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(collate_fn, img_size=img_size),
    )

    model = ObjectDetection(num_classes=10).to(device)
    if pretrained:
        model.load_backbone(pretrained, device)
    criterion = DetectionLoss(num_classes=num_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running = {"cls": 0.0, "box": 0.0, "dfl": 0.0}
        for images, gt_labels, gt_bboxes, mask_gt in loader:
            images = images.to(device)
            gt_labels, gt_bboxes, mask_gt = (
                gt_labels.to(device),
                gt_bboxes.to(device),
                mask_gt.to(device),
            )

            c3, c4, c5 = model.backbone(images)
            p3, p4, p5 = model.neck(c3, c4, c5)
            feats = [p3, p4, p5]
            preds = model.head(feats)

            loss_cls, loss_box, loss_dfl = criterion(
                preds, gt_labels, gt_bboxes, mask_gt
            )
            loss = 0.5 * loss_cls + 7.5 * loss_box + 1.5 * loss_dfl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["cls"] += loss_cls.item()
            running["box"] += loss_box.item()
            running["dfl"] += loss_dfl.item()

        scheduler.step()
        n = len(loader)
        print(
            f"epoch {epoch:3d} | lr {scheduler.get_last_lr()[0]:.5f} | "
            f"cls {running['cls']/n:.3f} box {running['box']/n:.3f} dfl {running['dfl']/n:.3f}"
        )

        torch.save(model.state_dict(), f"checkpoint_last.pth")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        epochs=50,
        device=device,
        pretrained=rf"./pretrained_model/pretrained_cnn_28.pth",
    )
