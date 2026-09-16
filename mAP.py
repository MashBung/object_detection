from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import ObjectDetectionDataset, collate_fn
from model import ObjectDetection
from util import bbox_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # 0.50, 0.55, ..., 0.95


@torch.no_grad()
def postprocess(preds, conf_thres=0.001, iou_thres=0.7, max_det=100):
    # preds: (B, 4+nc, A) -> 이미지마다 {"boxes", "scores", "labels"}
    results = []
    for p in preds.transpose(1, 2):  # (A, 4+nc)
        boxes, scores = p[:, :4], p[:, 4:]  # (A, 4), (A, nc)
        conf, cls = scores.max(dim=1)  # (A,), (A,)

        keep = conf > conf_thres
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]  # (M, 4), (M,), (M,)

        keep_idx = torchvision.ops.batched_nms(boxes, conf, cls, iou_thres)[:max_det]
        results.append(
            {
                "boxes": boxes[keep_idx],
                "scores": conf[keep_idx],
                "labels": cls[keep_idx],
            }
        )  # (K, 4), (K,), (K,)
    return results


def match_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels):
    """
    이미지 한 장에서 예측마다 IoU 기준 10단계 각각 TP인지 판정
    pred_boxes: (K, 4)   pred_labels: (K,)
    gt_boxes:   (G, 4)   gt_labels:   (G,)
    반환 tp: (K, 10) bool
    """
    K, G = pred_boxes.shape[0], gt_boxes.shape[0]
    tp = np.zeros((K, len(IOU_THRESHOLDS)), dtype=bool)
    if K == 0 or G == 0:
        return tp  # 예측이 없거나, GT가 없으면 전부 FP

    # (G, 1, 4) vs (1, K, 4) -> (G, K)  모든 GT × 예측 쌍의 IoU
    iou = bbox_iou(gt_boxes.unsqueeze(1), pred_boxes.unsqueeze(0)).squeeze(-1)
    same_cls = gt_labels.unsqueeze(1) == pred_labels.unsqueeze(0)  # (G, K)
    iou = (iou * same_cls).cpu().numpy()  # 클래스가 다르면 IoU 0

    for t, thr in enumerate(IOU_THRESHOLDS):
        gt_idx, pred_idx = np.nonzero(iou >= thr)  # 기준을 넘는 (GT, 예측) 쌍
        if len(gt_idx) == 0:
            continue

        # IoU 높은 쌍부터 짝지음
        order = np.argsort(-iou[gt_idx, pred_idx])
        gt_idx, pred_idx = gt_idx[order], pred_idx[order]

        # 예측 하나는 GT 하나에만, GT 하나는 예측 하나에만 (앞쪽 = IoU 높은 쌍 우선)
        # np.unique는 번호순으로 돌려주므로 np.sort로 IoU 높은 순서를 되살림
        _, first = np.unique(pred_idx, return_index=True)
        first = np.sort(first)
        gt_idx, pred_idx = gt_idx[first], pred_idx[first]
        _, first = np.unique(gt_idx, return_index=True)
        first = np.sort(first)
        pred_idx = pred_idx[first]

        tp[pred_idx, t] = True
    return tp


def compute_ap(recall, precision):
    """
    COCO 방식 101점 보간 AP
    recall, precision: (P,)  conf 높은 순으로 누적한 값
    """
    # precision을 오른쪽에서부터 누적 최댓값으로 (곡선을 계단형으로 펴기)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    rec_points = np.linspace(0, 1, 101)  # recall 0.00, 0.01, ..., 1.00
    idx = np.searchsorted(
        recall, rec_points, side="left"
    )  # 그 recall에 처음 도달하는 위치

    p_at_r = np.zeros(len(rec_points))
    valid = idx < len(recall)  # 도달하지 못한 recall 지점은 precision 0
    p_at_r[valid] = precision[idx[valid]]
    return p_at_r.mean()


def ap_per_class(tp, conf, pred_cls, gt_cls, eps=1e-16):
    """
    tp:       (P, 10)  데이터셋 전체 예측의 TP 여부
    conf:     (P,)     예측 점수
    pred_cls: (P,)     예측 클래스
    gt_cls:   (T,)     데이터셋 전체 GT 클래스
    반환: classes (C,), ap (C, 10)
    """
    order = np.argsort(-conf)  # conf 높은 순
    tp, pred_cls = tp[order], pred_cls[order]

    classes, n_gt = np.unique(gt_cls, return_counts=True)  # GT가 있는 클래스만 채점
    ap = np.zeros((len(classes), tp.shape[1]))

    for ci, c in enumerate(classes):
        m = pred_cls == c
        if m.sum() == 0:
            continue  # 이 클래스를 한 번도 예측 안 함 -> AP 0

        tpc = tp[m].cumsum(0)  # (Pc, 10) 누적 TP
        fpc = (~tp[m]).cumsum(0)  # (Pc, 10) 누적 FP
        recall = tpc / (n_gt[ci] + eps)  # 찾아낸 비율
        precision = tpc / (tpc + fpc)  # 맞힌 비율

        for t in range(tp.shape[1]):
            ap[ci, t] = compute_ap(recall[:, t], precision[:, t])

    return classes, ap


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_tp, all_conf, all_pred_cls, all_gt_cls = [], [], [], []

    for images, gt_labels, gt_bboxes, mask_gt in loader:
        preds = postprocess(model(images.to(device)))

        for i, p in enumerate(preds):
            m = mask_gt[i].squeeze(-1).bool()  # 패딩 제외, 진짜 GT만
            gt_boxes = gt_bboxes[i][m].to(device)  # (G, 4)
            gt_cls = gt_labels[i][m].squeeze(-1).long().to(device)  # (G,)

            all_tp.append(match_predictions(p["boxes"], p["labels"], gt_boxes, gt_cls))
            all_conf.append(p["scores"].cpu().numpy())
            all_pred_cls.append(p["labels"].cpu().numpy())
            all_gt_cls.append(gt_cls.cpu().numpy())

    classes, ap = ap_per_class(
        np.concatenate(all_tp),
        np.concatenate(all_conf),
        np.concatenate(all_pred_cls),
        np.concatenate(all_gt_cls),
    )
    return {
        "map": ap.mean(),  # IoU 0.5~0.95 평균, 클래스 평균
        "map_50": ap[:, 0].mean(),  # IoU 0.5
        "map_75": ap[:, 5].mean(),  # IoU 0.75
        "map_95": ap[:, 9].mean(),  # IoU 0.95
        "classes": classes,
        "ap_per_class": ap.mean(1),  # 클래스별 AP50-95
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ObjectDetection(num_classes=10).to(device)
    model.load_state_dict(torch.load("./ckpt/checkpoint_last.pth", map_location=device))

    dataset = ObjectDetectionDataset(
        r"C:\lnear_algebra\object_detection\data\images\val", img_size=320
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=partial(collate_fn, img_size=320),
    )

    result = evaluate(model, loader, device)
    print(f"mAP50-95: {result['map']:.4f}")
    print(f"mAP50   : {result['map_50']:.4f}")
    print(f"mAP75   : {result['map_75']:.4f}")
    print(f"mAP95   : {result['map_95']:.4f}")
    for c, a in zip(result["classes"], result["ap_per_class"]):
        print(f"  cls{c}: AP50-95 {a:.4f}")

"""
mAP50-95: 0.6269
mAP50   : 0.8120
mAP75   : 0.6856
mAP95   : 0.1135
  cls0: AP50-95 0.7983
  cls1: AP50-95 0.6786
  cls2: AP50-95 0.5038
  cls3: AP50-95 0.7146
  cls4: AP50-95 0.4400
  cls5: AP50-95 0.6217
  cls6: AP50-95 0.6384
  cls7: AP50-95 0.6421
  cls8: AP50-95 0.5817
  cls9: AP50-95 0.6497
"""
