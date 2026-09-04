import onnx
import numpy as np
import onnxruntime as ort

onnx.checker.check_model(onnx.load("object_detection.onnx"))

sess = ort.InferenceSession("object_detection.onnx", providers=["CPUExecutionProvider"])
