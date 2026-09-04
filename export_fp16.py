import onnx
from onnxconverter_common import float16

model_fp32 = onnx.load("object_detection.onnx")

model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)

onnx.save(model_fp16, "OD_fp16.onnx")
