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
