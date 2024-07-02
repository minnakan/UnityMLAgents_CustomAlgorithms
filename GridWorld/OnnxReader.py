import onnx

# Load the ONNX model
onnx_model_path = "C:/Users/minna/Documents/Repository/tensorboard/GridWorld.onnx"
onnx_model = onnx.load(onnx_model_path)

# Print the model's graph to understand its structure
onnx_model_graph = onnx.helper.printable_graph(onnx_model.graph)
print(onnx_model_graph)