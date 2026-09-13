import torch
import torch.nn as nn
import torch.nn.functional as F
from util import make_anchors, dist2bbox, bbox_iou
from tal import TaskAlignedAssigner


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.number_output = num_classes + 4 * reg_max
        self.reg_max = reg_max
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.strides = strides

    def decode_boxes(self, pred_dist, anc_points, stride_tensor):
        b, a, _ = pred_dist.shape  # (32,2100,_)
        dist = pred_dist.view(b, a, 4, self.reg_max).softmax(3)  # (32,2100,4,16)
        proj = torch.arange(self.reg_max, dtype=dist.dtype, device=dist.device)  # (16,)
        dist = dist @ proj  # (32,2100,4)
        boxes = dist2bbox(dist, anc_points, xywh=False, dim=-1)  # (32,2100,4)
        return boxes * stride_tensor  # (32,2100,4)

    def forward(self, preds, gt_labels, gt_bboxes, mask_gt):
        """
        preds: [(b, 74, 40, 40), (b, 74, 20, 20), (b, 74, 10, 10)]
        gt_labels: (b, max_gt, 1)
        gt_bboxes: (b, max_gt, 4)
        mask_gt: (b, max_gt, 1)
        """
        device = preds[0].device
        b = preds[0].shape[0]  # 32

        # (B, no, sum(HW)) 로 합치고 box/cls 분리
        pred_cat = torch.cat([p.view(b, self.number_output, -1) for p in preds], dim=2)
        # (b,74,2100)
        pred_dist, pred_scores = pred_cat.split(
            (4 * self.reg_max, self.num_classes), dim=1
        )
        """
        pred_dist: (32,64,2100)
        pred_scores: (32,10,2100)
        """

        pred_dist = pred_dist.permute(0, 2, 1).contiguous()  # (32,2100,64)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (32,2100,10)

        # anchor (픽셀 좌표) 준비
        anc_points, stride_tensor = make_anchors(preds, self.strides)
        """
        (2100,2), (2100,1)
        """

        anc_points_px = anc_points * stride_tensor  # 격자 -> 픽셀 (2100,2)

        # 예측 박스 디코딩
        pred_bboxes = self.decode_boxes(
            pred_dist, anc_points, stride_tensor
        )  # (32,2100,4)

        # --- 라벨 할당 ---
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),  # (32,2100,10)
            pred_bboxes.detach(),  # (32,2100,4)
            anc_points_px,  # (2100,2)
            gt_labels,  # (32, max_gt, 1)
            gt_bboxes,  # (32, max_gt, 4)
            mask_gt,  # (32, max_gt, 1)
        )
        """
        target_bboxes  (32, 2100, 4)     float    박스 좌표
        target_scores  (32, 2100, 10)    float    soft label [0, 1]
        fg_mask        (32, 2100)        bool     positive 여부
        """

        # --- cls 손실 (BCE) ---
        target_scores_sum = max(target_scores.sum(), 1)  # 스칼라
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        # 스칼라

        # --- box 손실 (CIoU) ---
        loss_box = torch.tensor(0.0, device=device)  # 스칼라
        loss_dfl = torch.tensor(0.0, device=device)  # 스칼라
        if fg_mask.any():  # fg_mask (32,2100) bool, N = positive 앵커 수
            pred_pos = pred_bboxes[fg_mask]  # (32,2100,4) -> (N,4)
            target_pos = target_bboxes[fg_mask]  # (32,2100,4) -> (N,4)
            weight = target_scores.sum(-1)[fg_mask]  # (32,2100,10) -> (32,2100) -> (N,)

            iou = bbox_iou(pred_pos, target_pos, xywh=False, CIoU=True).squeeze(-1)
            # bbox_iou -> (N,1), squeeze(-1) -> (N,)
            loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum
            # 스칼라

            # --- DFL ---
            # 정답 박스 -> 정답 거리(ltrb, 격자 단위)로 역변환
            target_ltrb = self.bboxdist(
                target_bboxes, anc_points_px, stride_tensor
            )  #  (32, 2100, 4)
            pred_dist_pos = pred_dist[fg_mask].view(
                -1, 4, self.reg_max
            )  # (2100, 4, 16)
            target_ltrb_pos = target_ltrb[fg_mask]  # (2100, 4)
            loss_dfl = (
                self.df_loss(pred_dist_pos, target_ltrb_pos).squeeze(-1) * weight
            ).sum() / target_scores_sum

        return loss_cls, loss_box, loss_dfl

    def df_loss(self, pred_dist, target_dist):
        # pred_dist:   (2100, 4, 16)  positive의 분포 로짓
        # target_dist: (2100, 4)           정답 거리 (격자 단위, 0~reg_max-1)
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
        return loss.mean(-1, keepdim=True)  # 4개 변 평균 -> (2100, 1)

    def bboxdist(self, bboxes, anc_points, stride_tensor):
        # 픽셀 박스 -> 격자 단위 거리 ltrb
        anc = anc_points / stride_tensor  # (2100,2)
        boxes = bboxes / stride_tensor  #  (32, 2100, 4)
        x1y1, x2y2 = boxes.chunk(2, dim=-1)
        lt = anc - x1y1
        rb = x2y2 - anc
        dist = torch.cat([lt, rb], dim=-1)
        return dist.clamp_(0, self.reg_max - 1 - 0.01)
