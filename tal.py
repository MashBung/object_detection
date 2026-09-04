import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    select_candidates_in_gts,
    get_alignment_metric,
    select_topk_candidates,
    select_highest_overlaps,
)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=10, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self, pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    ):
        """
        pred_scores: (b, 2100, 10)
        pred_bboxes: (b, 2100, 4)
        anc_points_px: (2100, 2)
        gt_labels: (b, max_gt, 1)
        gt_bboxes: (b,max_gt,4)
        mask_gt: (b,max_gt,1)
        """

        b, max_gt = gt_bboxes.shape[0], gt_bboxes.shape[1]

        if max_gt == 0:
            A = pred_scores.shape[1]
            return (
                torch.zeros_like(pred_scores),
                torch.zeros_like(pred_bboxes),
                torch.zeros(b, A, dtype=torch.bool, device=pred_scores.device),
            )

        # --- 1. 후보 거르기 ---
        mask_in_gts = select_candidates_in_gts(
            anc_points, gt_bboxes
        )  # (b, max_gt, 2100)
        mask_in_gts = mask_in_gts.bool() & mask_gt.bool()  # (b, max_gt, 2100) dtype = bool

        # --- 2. 정렬 점수 ---
        align_metric, overlaps = get_alignment_metric(
            pred_scores,
            pred_bboxes,
            gt_labels,
            gt_bboxes,
            mask_in_gts,
            self.alpha,
            self.beta,
        )  # (b, max_gt, 2100), (b, max_gt, 2100)

        # --- 3. top-k ---
        mask_topk = select_topk_candidates(
            align_metric, self.topk, mask_gt
        )  # (b, max_gt, 2100)
        mask_pos = mask_topk * mask_in_gts.to(mask_topk.dtype)

        # --- 4. 충돌 정리 ---
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, max_gt
        )

        # --- 5. 타깃 생성 ---
        target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # --- 6. 점수 정규화 ---
        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)  # GT별 최대 정렬점수
        pos_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # GT별 최대 IoU
        norm = (align_metric * pos_overlap / (pos_align + self.eps)).amax(
            dim=1
        )  # (b, 2100)
        target_scores = target_scores * norm.unsqueeze(-1)

        return target_bboxes, target_scores, fg_mask.bool()
        """
        target_bboxes  (b, 2100, 4)     float    박스 좌표
        target_scores  (b, 2100, 10)    float    soft label [0, 1]
        fg_mask        (b, 2100)        bool     positive 여부
        """

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        gt_labels: (b, max_gt, 1)
        gt_bboxes: (b, max_gt, 4)
        target_gt_idx: (b, A)
        fg_mask: (b, A)
        """
        b, max_gt = gt_labels.shape[0], gt_labels.shape[1]
        # A = target_gt_idx.shape[1]

        # 배치별 flat 인덱스로 GT를 gather
        batch_idx = torch.arange(b, device=gt_labels.device).view(-1, 1)
        flat_idx = target_gt_idx + batch_idx * max_gt  # (B, A)

        target_labels = gt_labels.long().flatten()[flat_idx]  # (B, A)
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]  # (B, A, 4)

        # 원핫 클래스, negative는 0으로
        target_scores = F.one_hot(target_labels, self.num_classes)  # (B, A, nc)
        target_scores = target_scores * fg_mask.unsqueeze(-1)  # negative 위치 제거

        return target_bboxes, target_scores
