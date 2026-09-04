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
