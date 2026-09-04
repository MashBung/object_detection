import torch
import torch.nn as nn
import math


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    feats:   헤드 입력 feature map 리스트 [p3, p4, p5]
    [(b, 74, 40, 40), (b, 74, 20, 20), (b, 74, 10, 10)]
    strides: 레벨별 stride (8, 16, 32)
    """
    anchor_points = []
    stride_tensor = []

    for feat, stride in zip(feats, strides):
        h, w = feat.shape[2], feat.shape[3]
        dtype, device = feat.dtype, feat.device
        # float32

        sx = torch.arange(w, dtype=dtype, device=device) + grid_cell_offset
        # [0.5, ... , shape[2] + 0.5]
        sy = torch.arange(h, dtype=dtype, device=device) + grid_cell_offset
        # [0.5, ... , shape[2] + 0.5]

        # indexing="ij" -> sy가 행(H), sx가 열(W)
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        # grid_y: (shape[2], shape[2])
        # grid_x: (shape[2], shape[2])

        # (H, W, 2) -> (H*W, 2), 각 행이 (x, y)
        anchor_points.append(torch.stack((grid_x, grid_y), dim=-1).view(-1, 2))
        # anchor_points: (2100, 2)

        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        # stride_tensor: (2100, 1)

    return torch.cat(anchor_points), torch.cat(stride_tensor)
    # anchor_points: (2100, 2)
    # stride_tensor: (2100, 1)




def dist2bbox(distance, anchor_point, xywh=False, dim=-1):
    """
    dist: (b, 2100, 4)
    distance:      ltrb 거리 (..., 4, ...)
    anchor_points: 격자 중심 좌표 (..., 2, ...)
    anchor_points: (2100, 2)
    """
    lt, rb = distance.chunk(2, dim)
    # lt: (b, 2100, 2)
    # rb: (b, 2100, 2)

    x1y1 = anchor_point - lt
    x2y2 = anchor_point + rb
    # x1y1: (b, 2100, 2)
    # x2y2: (b, 2100, 2)

    if xywh:
        center = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((center, wh), dim)
    return torch.cat((x1y1, x2y2), dim)
    # (b, 2100, 4)


def select_candidates_in_gts(anc_points, gt_bboxes, eps=1e-9):
    """
    anc_points: (2100, 2)
    gt_bboxes: (b,max_gt,4)
    """
    n_anchors = anc_points.shape[0]
    b, max_gt, _ = gt_bboxes.shape

    # 박스 좌상단/우하단으로 분리
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)
    # lt: (b * max_gt, 1, 2)
    # rb: (b * max_gt, 1, 2)

    # 네 방향 거리
    deltas = torch.cat([anc_points[None] - lt, rb - anc_points[None]], dim=2).view(
        b, max_gt, n_anchors, 4
    )
    # (b, max_gt, 2100, 4)
    """
    gt_(eps) — greater than, 즉 > eps 비교입니다.
    x.gt_(1e-9)   # x > 1e-9 → 불리언 텐서
    gt, lt, ge, le, eq 시리즈가 있고 각각 >, <, >=, <=, ==에 대응합니다.
    뒤의 밑줄 _은 in-place 연산이라는 표시입니다. 새 텐서를 만들지 않고 원본을 덮어써서 메모리를 절약하죠. add_, clamp_, mul_ 전부 같은 규칙입니다. 앞서 본 bbox2dist의 clamp_도 그렇고요.
    두 줄이 합쳐져서 하는 일이 "앵커가 GT 박스 안에 있나" 판정입니다.
    """
    return deltas.amin(dim=3).gt_(eps)  # (b, max_gt, 2100)


def bbox_iou(box1, box2, xywh=False, CIoU=False, eps=1e-7):
    """
    box1, box2: (..., 4)  브로드캐스팅 가능한 형태
    xywh=False -> xyxy 입력

    iou(Intersection over Union) = 교집합/합집합
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # 교집합: 겹치는 사각형의 넓이
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 합집합
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if not CIoU:
        return iou

    # 두 박스를 모두 감싸는 최소 사각형 (enclosing box)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    # 대각선^2, 중심거리^2 (외접 사각형 대각선의 제곱)
    c2 = cw**2 + ch**2 + eps
    # 두 중심 사이 거리의 제곱
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4

    # 종횡비 항
    v = (4 / math.pi**2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)  # CIoU


def get_alignment_metric(
    pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_in_gts, alpha=0.5, beta=6.0
):
    """
    pred_scores:   (b, 2100, 10)      예측 클래스 점수 (sigmoid 통과)
    pred_bboxes:   (b, 2100, 4)       예측 박스 xyxy
    gt_labels:   (b, max_gt, 1)  GT 클래스 인덱스
    gt_bboxes:   (b, max_gt, 4)  GT 박스
    mask_in_gts: (b, max_gt, 2100)  1단계 후보 마스크
    반환: align_metric (b, max_gt, 2100), overlaps (B, max_gt, 2100)
    """
    b, max_gt, _ = gt_bboxes.shape
    A = pred_scores.shape[1]

    bbox_scores = torch.zeros(
        b, max_gt, A, dtype=pred_scores.dtype, device=pred_scores.device
    )
    # bbox_scores: (b, max_gt, 2100)

    # 배치 인덱스와 GT 클래스 인덱스로 gather
    idx_b = torch.arange(b).view(-1, 1).expand(-1, max_gt)  # (b, max_gt)
    idx_c = gt_labels.squeeze(-1).long()  # (b, max_gt)
    # pred_scores[b, :, class] -> (B, max_gt, A)
    scores = pred_scores[idx_b, :, idx_c]  # (b,max_gt,A)

    bbox_scores[mask_in_gts] = scores[mask_in_gts]

    # --- IoU 항: 후보 위치에서만 계산 ---
    overlaps = torch.zeros(
        b, max_gt, A, dtype=pred_bboxes.dtype, device=pred_bboxes.device
    )
    # (B, max_gt, 1, 4) vs (B, 1, A, 4) -> (B, max_gt, A)
    iou = bbox_iou(gt_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1), CIoU=True).squeeze(
        -1
    )

    overlaps[mask_in_gts] = iou.clamp_(0)[mask_in_gts]

    # --- 정렬 점수 ---
    align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)

    return align_metric, overlaps


def select_topk_candidates(align_metric, topk=10, mask_gt=None):
    """
    align_metric: (b, max_gt, 2100)  정렬 점수
    mask_gt:      (b, max_gt, 1)  진짜 GT면 1, 패딩이면 0
    반환:          (b, max_gt, 2100)  top-k로 뽑힌 위치 마스크 (0/1)
    """
    topk_metric, topk_idxs = torch.topk(align_metric, topk, dim=-1)
    # (b, max_gt, 10), (b, max_gt, 10)

    mask_topk = torch.zeros_like(align_metric, dtype=torch.int8)
    mask_topk.scatter_(-1, topk_idxs, 1)

    # 후보 아닌 곳(점수 0)이 뽑혔으면 제거
    mask_topk = mask_topk * (align_metric > 0).to(torch.int8)

    # 패딩 GT 행 전체 0으로
    if mask_gt is not None:
        mask_topk = mask_topk * mask_gt.to(torch.int8)

    return mask_topk


def select_highest_overlaps(mask_pos, overlaps, max_gt):
    """
    mask_pos: (b, max_gt, 2100)  3단계까지의 positive 마스크
    overlaps: (b, max_gt, 2100)  2단계의 IoU 지도
    반환:
      target_gt_idx: (b, 2100)       각 anchor가 담당할 GT 인덱스
      fg_mask:       (B, 2100)       positive면 1
      mask_pos:      (b, max_gt, 2100)  충돌 정리된 마스크
    """

    # 각 anchor가 몇 개 GT에 뽑혔나 (GT축 합산)
    fg_mask = mask_pos.sum(dim=1)  # (b, 2100)

    if fg_mask.max() > 1:
        # 충돌 위치를 GT축으로 복제 (b, max_gt, 2100)
        mask_multi = (fg_mask.unsqueeze(1) > 1).expand(-1, max_gt, -1)

        # 그 anchor에서 IoU 최대인 GT 찾기
        max_overlaps_idx = overlaps.argmax(dim=1)  # (b, A)

        # 최대 IoU GT만 1인 one-hot
        max = torch.zeros_like(mask_pos)
        max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        # 충돌 위치는 is_max로 교체, 아닌 곳은 원래 mask 유지
        mask_pos = torch.where(mask_multi, max, mask_pos)
        fg_mask = mask_pos.sum(dim=1)

    # 최종: 각 anchor가 담당하는 GT 인덱스
    target_gt_idx = mask_pos.argmax(dim=1)

    return target_gt_idx, fg_mask, mask_pos
