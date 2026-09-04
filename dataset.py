import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms.v2 as tf
from PIL import Image


class ObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, img_size=320):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        exts = {".png"}
        self.img_paths = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in exts
        )

    def __len__(self):
        return len(self.img_paths)

    def label_path(self, img_path):
        parts = list(img_path.parts)
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --- 이미지 ---
        img = Image.open(img_path).convert("RGB")
        # img = img.resize((self.img_size, self.img_size))
        transform = tf.Compose(
            [
                tf.ToImage(),
                tf.ToDtype(torch.float32, scale=True),
            ]
        )
        img = transform(img)  # (3, H, W), 0~1

        # --- 라벨 ---
        label_path = self.label_path(img_path)
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, cx, cy, w, h = map(float, line.split())
                    boxes.append([cls, cx, cy, w, h])

        labels = (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5)
        )

        return img, labels


def collate_fn(batch, img_size=320):
    images, labels_list = zip(*batch)  # batch: [(img, labels), ...]

    # --- 1. 이미지 스택 ---
    images = torch.stack(images, dim=0)  # (B, 3, H, W)

    b = len(labels_list)
    max_gt = max(len(label) for label in labels_list)
    max_gt = max(max_gt, 1)

    # --- 2. 패딩 컨테이너 ---
    gt_labels = torch.zeros(b, max_gt, 1)
    gt_bboxes = torch.zeros(b, max_gt, 4)
    mask_gt = torch.zeros(b, max_gt, 1)

    for i, labels in enumerate(labels_list):
        n = len(labels)
        if n == 0:
            continue

        cls = labels[:, 0:1]
        cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]

        # --- 3. 정규화 xywh -> 픽셀 xyxy ---
        x1 = (cx - w / 2) * img_size
        y1 = (cy - h / 2) * img_size
        x2 = (cx + w / 2) * img_size
        y2 = (cy + h / 2) * img_size
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (n, 4)

        gt_labels[i, :n] = cls
        gt_bboxes[i, :n] = boxes
        mask_gt[i, :n] = 1.0

    return images, gt_labels, gt_bboxes, mask_gt


if __name__ == "__main__":
    dataset = ObjectDetectionDataset("data/images/train", img_size=320)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, img_size=320),
    )

    images, gt_labels, gt_bboxes, mask_gt = next(iter(loader))

    print("images   :", images.shape)  # (4, 3, 320, 320)
    print("gt_labels:", gt_labels.shape)  # (4, max_gt, 1)
    print("gt_bboxes:", gt_bboxes.shape)  # (4, max_gt, 4)
    print("mask_gt  :", mask_gt.shape)

    # 첫 이미지의 진짜 GT만 출력
    n = int(mask_gt[0].sum())
    print(f"\n이미지0 객체 수: {n}")
    print("박스(픽셀 xyxy):\n", gt_bboxes[0, :n])
    print("클래스:", gt_labels[0, :n].squeeze(-1).tolist())

    valid = gt_bboxes[mask_gt.squeeze(-1).bool()]
    print("\nx1<x2:", bool((valid[:, 0] < valid[:, 2]).all()))
    print("y1<y2:", bool((valid[:, 1] < valid[:, 3]).all()))
    print("범위 0~320:", bool((valid >= 0).all() and (valid <= 320).all()))
