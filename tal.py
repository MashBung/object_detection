import torch
import torch.nn as nn
import torch.nn.functional as F
from util import (
    select_candidates_in_gts,
    get_alignment_metric,
    select_topk_candidates,
    select_highest_overlaps,
)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=10, alpha=0.5, beta=5.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self, pred_scores, pred_bboxes, anc_points_px, gt_labels, gt_bboxes, mask_gt
    ):
        """
        pred_scores: (32,2100,10)
        pred_bboxes: (32,2100,4)
        anc_points: (2100, 2)
        gt_labels: (32, max_gt, 1)
        gt_bboxes: (32,max_gt,4)
        mask_gt: (32,max_gt,1)
        """

        b, max_gt = gt_bboxes.shape[0], gt_bboxes.shape[1]

        if max_gt == 0:
            A = pred_scores.shape[1]
            return (
                torch.zeros_like(pred_bboxes),
                torch.zeros_like(pred_scores),
                torch.zeros(b, A, dtype=torch.bool, device=pred_scores.device),
            )

        # --- 1. 후보 거르기 ---
        mask_in_gts = select_candidates_in_gts(
            anc_points_px, gt_bboxes
        )  # (32,max_gt,2100)
        mask_in_gts = mask_in_gts.bool() & mask_gt.bool()  # dtype=bool

        # --- 2. 정렬 점수 ---
        align_metric, overlaps = get_alignment_metric(
            pred_scores,  # (32,2100,10)
            pred_bboxes,  # (32,2100,4)
            gt_labels,  # (32, max_gt, 1)
            gt_bboxes,  # (32,max_gt,4)
            mask_in_gts,  # (32,max_gt,2100)
            self.alpha,  # 0.5
            self.beta,  # 5.0
        )
        """
        align_metric: (32, max_gt, 2100)
        overlaps: (32, max_gt, 2100)
        """

        # --- 3. top-k ---
        mask_topk = select_topk_candidates(
            align_metric, self.topk, mask_gt
        )  # (32, max_gt, 2100)
        mask_pos = mask_topk * mask_in_gts.to(mask_topk.dtype)  # (32, max_gt, 2100)

        # --- 4. 충돌 정리 ---
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, max_gt
        )
        """
        target_gt_idx: (32, 2100)
        fg_mask: (32, 2100)
        mask_pos: (32,max_gt,2100)
        """

        # --- 5. 타깃 생성 ---
        target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        # (32, 2100, 4)
        # (32, 2100, 10)

        # --- 6. 점수 정규화 ---
        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)  # GT별 최대 정렬점수
        pos_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # GT별 최대 IoU
        norm = (align_metric * pos_overlap / (pos_align + self.eps)).amax(
            dim=1
        )  # (B, A)
        target_scores = target_scores * norm.unsqueeze(-1)

        return target_bboxes, target_scores, fg_mask.bool()
        """
        target_bboxes  (32, 2100, 4)     float    박스 좌표
        target_scores  (32, 2100, 10)    float    soft label [0, 1]
        fg_mask        (32, 2100)        bool     positive 여부
        """

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        gt_labels: (32, max_gt, 1)
        gt_bboxes: (32, max_gt, 4)
        target_gt_idx: (32, 2100)
        fg_mask: (32, 2100)
        """

        b, max_gt = gt_labels.shape[0], gt_labels.shape[1]
        # 32, max_gt

        # 앵커마다 담당 GT의 라벨을 가져옴
        # dim=1(GT 축)에서 target_gt_idx 번호의 GT를 골라옴
        labels = gt_labels.long().squeeze(-1)  # (32, max_gt)
        target_labels = torch.gather(
            input=labels, dim=1, index=target_gt_idx
        )  # (32, 2100)

        # 앵커마다 담당 GT의 박스를 가져옴
        # 좌표 4개를 통째로 가져오도록 index를 마지막 축으로 4칸 늘림
        idx_box = target_gt_idx.unsqueeze(-1).expand(-1, -1, 4)  # (32, 2100, 4)
        target_bboxes = torch.gather(
            input=gt_bboxes, dim=1, index=idx_box
        )  # (32, 2100, 4)

        # 원핫 클래스, negative는 0으로
        target_scores = F.one_hot(target_labels, self.num_classes)  # (32, 2100, 10)
        target_scores = target_scores * fg_mask.unsqueeze(-1)  # (32, 2100, 10)

        return target_bboxes, target_scores
        # (32, 2100, 4)
        # (32, 2100, 10)
