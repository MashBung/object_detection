import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

from model import ObjectDetection


@torch.no_grad()
def detect(
    model, image_path, img_size=320, conf_thres=0.25, iou_thres=0.45, device="cpu"
):
    model.eval()

    # --- 이미지 전처리 (학습 때와 동일하게) ---
    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    img = pil.resize((img_size, img_size))
    x = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # --- forward: eval 모드라 디코딩된 박스가 나옴 ---
    preds = model(x)  # (1, 4+nc, A)
    preds = preds[0].transpose(0, 1)  # (A, 4+nc)

    boxes = preds[:, :4]  # xyxy 픽셀 (320 기준)
    scores = preds[:, 4:]  # (A, nc) 클래스별 점수

    # --- 각 위치의 최고 클래스와 점수 ---
    conf, cls = scores.max(dim=1)  # (A,), (A,)

    # --- 1. conf threshold로 거르기 ---
    keep = conf > conf_thres
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    if boxes.shape[0] == 0:
        return []  # 검출 없음

    # --- 2. 클래스별 NMS ---
    keep_idx = torchvision.ops.batched_nms(boxes, conf, cls, iou_thres)
    boxes, conf, cls = boxes[keep_idx], conf[keep_idx], cls[keep_idx]

    # --- 3. 원본 이미지 크기로 좌표 복원 ---
    scale_x, scale_y = orig_w / img_size, orig_h / img_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # 결과 리스트로 정리
    results = []
    for box, c, k in zip(boxes, conf, cls):
        results.append(
            {
                "box": box.tolist(),
                "conf": c.item(),
                "cls": int(k.item()),
            }
        )
    return results


def draw(image_path, results, save_path="result_pth_fp32.jpg"):
    pil = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for r in results:
        x1, y1, x2, y2 = r["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 12), f"cls{r['cls']} {r['conf']:.2f}", fill="red")
    pil.save(save_path)
    print("저장:", save_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = 10

    model = ObjectDetection(num_classes=nc).to(device)
    model.load_state_dict(torch.load("checkpoint_last.pth", map_location=device))

    image_path = rf"./bear.png"  # 실제 파일명으로
    results = detect(model, image_path, conf_thres=0.15, device=device)

    print(f"검출 {len(results)}개")
    for r in results:
        print(
            f"  cls{r['cls']} conf {r['conf']:.3f} box {[round(v,1) for v in r['box']]}"
        )

    draw(image_path, results)
