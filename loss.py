import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_anchors, dist2bbox, bbox_iou
from tal import TaskAlignedAssigner


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.number_output = num_classes + 4 * reg_max
        self.strides = strides
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)

    def forward(self, preds, gt_labels, gt_bboxes, mask_gt):
        """
        preds: [(b, 74, 40, 40), (b, 74, 20, 20), (b, 74, 10, 10)]
        feats:
        gt_labels: (b, max_gt, 1)
        gt_bboxes: (b, max_gt, 4)
        mask_gt: (b, max_gt, 1)
        """
        device = preds[0].device
        b = preds[0].shape[0]

        # (B, no, sum(HW)) 로 합치고 box/cls 분리
        pred_cat = torch.cat([p.view(b, self.number_output, -1) for p in preds], dim=2)
        # (b, 74, 2100)
        pred_dist, pred_scores = pred_cat.split(
            (4 * self.reg_max, self.num_classes), dim=1
        )
        # pred_dist: (b, 64, 2100)
        # pred_scores: (b, 10, 2100)
        
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        # pred_dist: (b, 2100, 64)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # pred_scores: (b, 2100,10)

        # anchor (픽셀 좌표) 준비
        anc_points, stride_tensor = make_anchors(preds, self.strides)
        # anchor_points: (2100, 2)
        # stride_tensor: (2100, 1)

        anc_points_px = anc_points * stride_tensor  # 격자 -> 픽셀
        # anc_points_px: (2100, 2)

        # 예측 박스 디코딩 (TAL에 넘길 픽셀 박스)
        pred_bboxes = self.decode_boxes(pred_dist, anc_points, stride_tensor)
        # pred_bboxes: (b, 2100, 4)

        # --- 라벨 할당 ---
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anc_points_px,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        """
        target_bboxes  (B, A, 4)     float    박스 좌표
        target_scores  (B, A, nc)    float    soft label [0, 1]
        fg_mask        (B, A)        bool     positive 여부
        """

        # --- cls 손실 (BCE) ---
        target_scores_sum = max(target_scores.sum(), 1)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        # --- box 손실 (CIoU) ---
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        if fg_mask.any():
            pred_pos = pred_bboxes[fg_mask]
            target_pos = target_bboxes[fg_mask]
            weight = target_scores.sum(-1)[fg_mask]

            iou = bbox_iou(pred_pos, target_pos, xywh=False, CIoU=True).squeeze(-1)
            loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

            # --- DFL ---
            # 정답 박스 -> 정답 거리(ltrb, 격자 단위)로 역변환
            target_ltrb = self.bboxdist(target_bboxes, anc_points_px, stride_tensor)
            pred_dist_pos = pred_dist[fg_mask].view(-1, 4, self.reg_max)
            target_ltrb_pos = target_ltrb[fg_mask]
            loss_dfl = (
                self.df_loss(pred_dist_pos, target_ltrb_pos).squeeze(-1) * weight
            ).sum() / target_scores_sum

        return loss_cls, loss_box, loss_dfl

    def df_loss(self, pred_dist, target_dist):
        # pred_dist:   (P, 4, reg_max)  positive의 분포 로짓
        # target_dist: (P, 4)           정답 거리 (격자 단위, 0~reg_max-1)
        tl = target_dist.long()  # 왼쪽 bin (내림)
        tr = tl + 1  # 오른쪽 bin
        wl = tr - target_dist  # 왼쪽 가중 (4 - 3.7 = 0.3)
        wr = target_dist - tl  # 오른쪽 가중 (3.7 - 3 = 0.7)

        loss = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max), tl.view(-1), reduction="none"
            ).view(tl.shape)
            * wl
            + F.cross_entropy(
                pred_dist.view(-1, self.reg_max), tr.view(-1), reduction="none"
            ).view(tr.shape)
            * wr
        )
        return loss.mean(-1, keepdim=True)  # 4개 변 평균 -> (P, 1)

    def bboxdist(self, bboxes, anc_points, stride_tensor):
        # 픽셀 박스 -> 격자 단위 거리 ltrb
        anc = anc_points / stride_tensor
        boxes = bboxes / stride_tensor
        x1y1, xyy2 = boxes.chunk(2, dim=-1)
        lt = anc - x1y1
        rb = xyy2 - anc
        dist = torch.cat([lt, rb], dim=-1)
        return dist.clamp_(0, self.reg_max - 1 - 0.01)

    def decode_boxes(self, pred_dist, anc_points, stride_tensor):
        """
        # pred_dist: (b, 2100, 64)
        # anchor_points: (2100, 2)
        # stride_tensor: (2100, 1)
        """
        b, a, _ = pred_dist.shape
        # b: b
        # a: 2100
        dist = pred_dist.view(b, a, 4, self.reg_max).softmax(3)  
        # (b, 2100, 4, 16)
        proj = torch.arange(self.reg_max, dtype=dist.dtype, device=dist.device)  # [16]
        # [0, ..., 15]
        dist = dist @ proj  # (b, 2100, 4)
        boxes = dist2bbox(dist, anc_points, xywh=False, dim=-1)
        # boxes = (b, 2100, 4)
        return boxes * stride_tensor
        # (b, 2100, 4)
    

