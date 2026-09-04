import torch
import numpy as np
import onnxruntime as ort
from model import ObjectDetection

model = ObjectDetection(num_classes=10)
model.load_state_dict(torch.load("./checkpoint_last.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    torch_out = model(dummy)


def compare(path, tag):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": dummy.numpy()})
    for i, (t, o) in enumerate(zip(torch_out, outs)):
        t = t.numpy()
        abs_err = np.abs(t - o).max()
        rel_err = abs_err / np.abs(t).max()
        print(
            tag,
            i,
            t.shape,
            np.abs(t).max(),
            abs_err,
            rel_err,
            np.isinf(o).any(),
            np.isnan(o).any(),
        )


compare("object_detection.onnx", "fp32")
compare("OD_fp16.onnx", "fp16")
