import onnx
import torch
import onnxruntime as ort
from onnxconverter_common import float16
from model import ObjectDetection

ckpt = "./ckpt/checkpoint_last.pth"
onnx_fp32 = "./ckpt/model_fp32.onnx"
onnx_fp16 = "./ckpt/model_fp16.onnx"


def load_model():
    model = ObjectDetection(num_classes=10)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def export_fp32(model):
    dummy = torch.zeros(1, 3, 320, 320)

    torch.onnx.export(
        model,
        dummy,
        onnx_fp32,
        input_names=["images"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_fp32))


def convert_fp16():
    model_fp32 = onnx.load(onnx_fp32)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, onnx_fp16)


def compare(model):
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        ref = model(x).numpy()  # (1, 14, 2100)

    for path in [onnx_fp32, onnx_fp16]:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"images": x.numpy()})[0]
        box_diff = abs(out[:, :4] - ref[:, :4]).max()  # 픽셀
        score_diff = abs(out[:, 4:] - ref[:, 4:]).max()  # 확률
        print(
            f"{path}: shape {out.shape}, "
            f"box 최대오차 {box_diff:.4f}px, score 최대오차 {score_diff:.6f}"
        )


if __name__ == "__main__":
    model = load_model()
    print(model)
    export_fp32(model)
    convert_fp16()
    compare(model)
