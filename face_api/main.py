import sys
import os
import time
from fastapi import FastAPI, Response, Form, UploadFile
from fastapi import Request, File, status
import json
from face_recognition import RecognitionTriton
from detector import Detector
from utils import *
import settings
from json import JSONEncoder
import requests
import asyncio
import tritonclient.grpc as tritongrpclient
import logging
import numpy as np
import cv2

app = FastAPI()


triton_client = tritongrpclient.InferenceServerClient(url=settings.TRITON_SERVER_URL, verbose=False)
recognizer = RecognitionTriton(triton_client=triton_client, model_name=settings.RECOGNIZER_MODEL_NAME,
                                   model_version=settings.RECOGNIZER_MODEL_VERSION, batch_size=settings.RECOGNIZER_BATCH)

detector = Detector(weight_path=settings.DETECTION_WEIGHT_PATH, cfg_mnet=cfg_mnet)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def readb64(uri):
    nparr = np.fromstring(uri, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post("/")
async def main():
    return {"success": True}

@app.post("/api/get_face_embedding/", status_code=200)
async def get_face_embedding(request: Request, imageFile: UploadFile = File(...), device_id: str = Form()):
    if imageFile.content_type is not None:
        image_base64 = await imageFile.read()
        filename = imageFile.filename
        image = readb64(image_base64)
        image = recognizer_image_preprocess(image)
        if (image.shape[2] == 112 and image.shape[3] == 112):
            t1 = time.time()
            feature = recognizer.predict(image)
            t2 = time.time()
            if feature is not None:
                encodedNumpyData = json.dumps(feature, cls=NumpyArrayEncoder)
                message = {'status': True, 'filename': filename, 'embedding': encodedNumpyData, 'device_id': device_id,
                           'exec_time': t2-t1}
                # data = requests.post(url=settings.RESULT_SEND_URL, data=message)
                return message
            else:
                return {'status': False, 'message': 'tritonserver is unavailable'}
        else:
            return {'status': False, 'message': 'wrong image shape'}
    else:
        return {'status': False, 'message': 'image not found'}


@app.post("/api/compare_two_face/", status_code=200)
async def compare_two_face(request: Request, face1: UploadFile = File(...), face2: UploadFile = File(...)):
    if face1.content_type is not None and face2.content_type is not None:
        face1_base64 = await face1.read()
        face2_base64 = await face2.read()
        faceImg1 = readb64(face1_base64)
        faceImg2 = readb64(face2_base64)
        cropFace1 = detector.detect(faceImg1, confidence_threshold=settings.DETECTION_CONF_THRESHOLD, nms_threshold=settings.DETECTION_NMS_THRESHOLD)
        cropFace2 = detector.detect(faceImg2, confidence_threshold=settings.DETECTION_CONF_THRESHOLD, nms_threshold=settings.DETECTION_NMS_THRESHOLD)
        if cropFace1 is not None and cropFace2 is not None:
            img1 = recognizer_image_preprocess(cropFace1)
            img2 = recognizer_image_preprocess(cropFace2)
            featureFace1 = recognizer.predict(img1)
            featureFace2 = recognizer.predict(img2)
            similarity = round(float(np.dot(featureFace1, featureFace2.T))*100, 2)
            return {'status': True, 'similarity': similarity}
        else:
            return {'status': False, 'message': 'face not found'}
    else:
        return {'status': False, 'message': 'images not found'}
