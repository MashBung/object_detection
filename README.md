# object_detection

사전학습한 CNN백본 위에, **YOLOv8 구조(PAN_FPN, neck, anchor-free head, Task-Aligned Assigner, DFL loss)를 Pytorch로 구현한 객체 탐지기입니다.**

fp32

<img width="634" height="442" alt="result_pth_fp32" src="https://github.com/user-attachments/assets/6e8816b9-8a1e-46aa-ac96-29069cd28550" />

fp16

<img width="634" height="442" alt="result_onnx_fp16" src="https://github.com/user-attachments/assets/ea37f369-ec58-4424-a92a-5077121658c2" />


## 핵심 결과
 
| 항목 | 값 |
|---|---|
| mAP@0.5 | 0.7331 (50 epochs) |
| 클래스 수 | 10 |
| 입력 해상도 | 320 × 320 |
| 앵커 수 | 2100 (40² + 20² + 10²) |
| 파라미터 수 | 17.91M |
| 학습 환경 | RTX 5070 Ti |

## 왜 API호출이 아니고 구현했나
- 한때 sota 모델이라 부르던 YOLOv8이 어떻게 작동하는지 이해하기 위해서입니다. 신경망, loss와 정답과 예측를 처리하는 구현하지 않으면 블랙박스로 남습니다.
- 자체 사전학습한 백본을 그대로 연결하려면 채널과 stride에 맞춰 neck·head를 직접 조립해야 했습니다.


## 전체 구조
 
```
Input 3×320×320
 │
 ├─ Backbone (backbone.py) — 500-class 사전학습 신경망, classifier 제거
 │     C3: (B,128,40,40) stride 8
 │     C4: (B,256,20,20) stride 16
 │     C5: (B,512,10,10) stride 32
 │
 ├─ Neck (neck.py) — PAN-FPN
 │     C5 → SPPF → top-down(upsample + concat + C2f) → bottom-up(stride-2 Conv + concat + C2f)
 │     P3: (B,128,40,40)   P4: (B,256,20,20)   P5: (B,512,10,10)
 │
 └─ Head (head.py) — Decoupled anchor-free Detect
       reg branch: Conv×2 → 1×1 Conv → 4×16 bins (DFL)
       cls branch: Conv×2 → 1×1 Conv → 10 classes
       train: 레벨별 raw 출력 [(B,74,40,40), (B,74,20,20), (B,74,10,10)]
       eval : DFL 적분 → dist2bbox → (B, 4+10, 2100)
```

### 구성 요소
 
| 파일 | 내용 |
|---|---|
| `blocks.py` | `Conv`(Conv+BN+SiLU), `Bottleneck`, `C2f`(split → bottleneck 누적 → concat), `SPPF`, `DFL`(16-bin softmax 기댓값 적분) |
| `neck.py` | PAN-FPN. `F.interpolate(size=...)`로 해상도 불일치에 안전하게 top-down / bottom-up 융합 |
| `head.py` | reg/cls 분리 head, 사전확률 기반 `bias_init`, train/eval 분기, 입력 해상도 변경 시에만 anchor 재생성 |
| `tal.py` | Task-Aligned Assigner — GT 내부 후보 필터 → alignment metric(`s^α · IoU^β`) → top-k → 충돌 정리 → soft target 생성 |
| `loss.py` | BCE(cls) + CIoU(box) + DFL(reg), positive 가중치를 target score 합으로 정규화 |
| `utils.py` | `make_anchors`, `dist2bbox`, `bbox_iou`(CIoU), TAL 부품 함수 4개 |
| `dataset.py` | YOLO 포맷(`cls cx cy w h`) 파싱, 가변 길이 GT를 `max_gt`로 패딩하는 `collate_fn` |
| `model.py` | backbone + neck + head 조립, `load_backbone`(classifier 키 제거 후 로드), gradient 도달 검증 |

## 학습 설정
 
`train.py`
 
| 항목 | 설정 |
|---|---|
| Backbone init | 500-class 사전학습 weight (`pretrained_cnn_28.pth`), 전체 fine-tuning |
| Optimizer | AdamW (lr=1e-3, weight_decay=5e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss weight | 0.5·cls + 7.5·box + 1.5·dfl (YOLOv8 기본값) |
| Batch size | 32 |
| Epochs | 50 |
| Assigner | TAL (topk=10, α=0.5, β=6.0) |

## 구현하면서 정리한 핵심 개념
 
**TAL(Task-Aligned Assigner)은 왜 필요한가**
anchor-free head는 2100개 위치 전부에서 예측을 내놓지만, 그중 어떤 위치가 어떤 GT를 "책임질지"는 정해져 있지 않습니다. TAL은 분류 점수와 IoU를 함께 본 alignment metric으로 GT마다 상위 k개 위치를 positive로 고르고, 여러 GT에 겹친 위치는 IoU가 가장 높은 GT에 배정합니다. 결과로 나오는 target score는 0/1이 아닌 **soft label**이어서 잘 정렬된 예측에 더 큰 가중치가 실립니다.
 
**DFL(Distribution Focal Loss)은 왜 거리를 분포로 예측하나**
박스의 각 변까지의 거리를 하나의 실수로 회귀하면 경계가 애매한 객체에서 불안정합니다. 대신 0~15 bin에 대한 확률 분포를 예측하고 기댓값으로 거리를 얻으면, 정답 거리 3.7은 bin 3에 0.3, bin 4에 0.7의 가중치를 주는 cross-entropy로 학습할 수 있어 더 안정적입니다.
 
**좌표계 일치**
TAL과 CIoU는 GT·예측 박스와 anchor를 **모두 픽셀 좌표**로 받아야 합니다. anchor는 격자 단위로 생성되므로 `anc_points * stride`로 변환해서 넘겨야 하는데, 이 부분을 놓치면 positive가 거의 안 잡히면서 loss가 내려가지 않습니다.
