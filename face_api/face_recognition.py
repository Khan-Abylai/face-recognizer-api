import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import cv2
from sklearn import preprocessing
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def parse_model_grpc(model_metadata, model_config):
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    return (model_config.max_batch_size, input_metadata.name, output_metadata.name, input_metadata.datatype)


class RecognitionTriton(object):
    def __init__(self, triton_client, model_name, model_version, batch_size, async_set=False, streaming=False):
        self.triton_client = triton_client
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size
        self.async_set = async_set
        self.streaming = streaming
        self.model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
        self.model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
        self.max_batch_size, self.input_name, self.output_name, self.dtype = parse_model_grpc(
            self.model_metadata, self.model_config.config)
        if self.max_batch_size == 0:
            self.max_batch_size = 1

    def predict(self, x: np.ndarray):
        input = grpcclient.InferInput(self.input_name, (self.max_batch_size, 3, 112, 112), 'FP32')
        input.set_data_from_numpy(x)

        output = grpcclient.InferRequestedOutput(self.output_name)
        try:
            response = self.triton_client.infer(self.model_name, model_version=self.model_version,
                                                inputs=[input], outputs=[output])
            out = response.as_numpy(self.output_name)
            out = np.asarray(out, dtype=np.float32)
            emb_n = preprocessing.normalize(out.reshape(1, -1)).flatten()
            return emb_n
        except InferenceServerException as e:
            print("Error: ", e)
            return None

# class Recognition():
#     __ENGINE_PATH = "./model_face.engine"
#
#     def __init__(self):
#         self.cfx = cuda.Device(0).make_context()
#         self.stream = cuda.Stream()
#         TRT_LOGGER = trt.Logger(trt.Logger.INFO)
#         runtime = trt.Runtime(TRT_LOGGER)
#
#         with open(Recognition.__ENGINE_PATH, "rb") as f:
#             engine = runtime.deserialize_cuda_engine(f.read())
#         self.context = engine.create_execution_context()
#
#         host_inputs = []
#         cuda_inputs = []
#         host_outputs = []
#         cuda_outputs = []
#         bindings = []
#
#         for binding in engine:
#             size = trt.volume(engine.get_binding_shape(binding)) * 1
#             dtype = trt.nptype(engine.get_binding_dtype(binding))
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             cuda_mem = cuda.mem_alloc(host_mem.nbytes)
#             bindings.append(int(cuda_mem))
#             if engine.binding_is_input(binding):
#                 host_inputs.append(host_mem)
#                 cuda_inputs.append(cuda_mem)
#             else:
#                 host_outputs.append(host_mem)
#
#                 cuda_outputs.append(cuda_mem)
#
#         self.engine = engine
#         self.host_inputs = host_inputs
#         self.cuda_inputs = cuda_inputs
#         self.host_outputs = host_outputs
#         self.cuda_outputs = cuda_outputs
#         self.bindings = bindings
#
#     def get_emdbedding(self, x: np.ndarray):
#         self.cfx.push()
#         stream = self.stream
#         context = self.context
#         host_inputs = self.host_inputs
#         cuda_inputs = self.cuda_inputs
#         host_outputs = self.host_outputs
#         cuda_outputs = self.cuda_outputs
#         bindings = self.bindings
#         image = np.ascontiguousarray(x)
#         np.copyto(host_inputs[0], image.ravel())
#         cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
#         context.execute_async(bindings=bindings, stream_handle=stream.handle)
#         cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
#         stream.synchronize()
#
#         self.cfx.pop()
#
#         output = host_outputs[0]
#
#         return output
#
#     def destroy(self):
#         # Remove any context from the top of the context stack, deactivating it.
#         self.cfx.pop()
