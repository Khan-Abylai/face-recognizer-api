# face-recognizer-api


## Getting started
### 1. Convert onnx file to tensorrt plan file with the command:
#### trtexec --onnx=./face_api/models/model_r100.onnx --saveEngine=./triton/models/face_trt/1/model.plan 
