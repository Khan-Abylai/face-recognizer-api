import sys
sys.path.append('..')
import tritonclient.grpc as tritonhttpclient
import cv2
import numpy as np
import torch
import time
from utils import *
from sklearn import preprocessing

VERBOSE = False
input_name = 'data'
input_shape = (1, 3, 112, 112)
input_dtype = 'FP32'
output_name = ['output']
model_name = 'face_trt'
url = '10.66.100.20:8001'
model_version = '1'

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

print(model_metadata)
print(model_config)
img_path1 = "../images/Arman1_crop.jpg"
img = cv2.imread(img_path1)
img = recognizer_image_preprocess(img)

input0 = tritonhttpclient.InferInput(input_name, (1, 3, 112, 112), 'FP32')
input0.set_data_from_numpy(img)

t1 = time.time()
output1 = tritonhttpclient.InferRequestedOutput(output_name[0])
response = triton_client.infer(model_name, model_version=model_version,
                               inputs=[input0], outputs=[output1])
t2 = time.time()
out1 = response.as_numpy('output')
out1 = np.asarray(out1, dtype=np.float32)
emb1_n = out1.reshape(1, -1)
emb1_n = preprocessing.normalize(emb1_n).flatten()

print(emb1_n)
print("triton exec:", time.time()-t1)