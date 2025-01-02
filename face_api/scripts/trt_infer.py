import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import torch
from sklearn import preprocessing
# from numba import cuda


TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def image_preprocess(image_path):
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img

class TrtModel():

    def __init__(self, engine_file_path):

        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings


    def predict(self,x:np.ndarray, batch_size):

        self.cfx.push()

        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        image = np.ascontiguousarray(x)
        np.copyto(host_inputs[0], image.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()

        self.cfx.pop()

        output = host_outputs[0]

        return output

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        pass


if __name__ == "__main__":
    img_path1 = "images/Arman5_crop.jpg"
    img1 = image_preprocess(img_path1)
    trt_engine_path = "model_face.engine"
    model = TrtModel(trt_engine_path)
    t1 = time.time()
    emb1 = model.predict(img1.numpy(), batch_size=1)
    t2 = time.time()
    emb1_n = emb1.reshape(1, -1)
    emb1_n = preprocessing.normalize(emb1_n).flatten()
    print(f"infer trt time: {t2 - t1}")
    print(emb1_n)
    # result = np.dot(emb1_n, emb2_n.T)
    # print("similarity trt:", result)
    model.destroy()


