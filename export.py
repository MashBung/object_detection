import torch
from model import ObjectDetection

model = ObjectDetection(num_classes=10)
model.load_state_dict(torch.load("./checkpoint_last.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 320, 320)

torch.onnx.export(
    model,
    dummy,
    "object_detection.onnx",
    input_names=["images"],
    output_names=["p3", "p4", "p5"],  # head가 리스트로 반환하면 개수 맞춰서
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,  # 배치/해상도 전부 고정
)


import onnx
import numpy as np
import onnxruntime as ort

onnx.checker.check_model(onnx.load("object_detection.onnx"))

sess = ort.InferenceSession("object_detection.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"images": dummy.numpy()})

with torch.no_grad():
    torch_out = model(dummy)

for t, o in zip(torch_out, ort_out):
    print(np.abs(t.numpy() - o).max())
