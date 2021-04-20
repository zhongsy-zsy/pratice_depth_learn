import onnx

model = onnx.load('gpu.onnx')
onnx.checker.check_model(model)
print("====> pass")