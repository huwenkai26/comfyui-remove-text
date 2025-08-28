import onnxruntime as rt
import numpy as np
import uuid
import cv2
from decode import SegDetectorRepresenter
from fastapi import FastAPI, Request
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import requests
import uvicorn, json

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
app = FastAPI()


class RemoveTextRequst(BaseModel):
    originImages: List[str]


def textRemove(oriImg):
    img = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    box_list, score_list = text_handle.process(img, 960)

    h, w, c = img.shape
    print("w - h", w, h)
    # Fill temp image with black
    img_inpaint = img.copy()
    img_temp = cv2.rectangle(img, [0, 0], [w, h], (0, 0, 0), -1)

    print(box_list)
    # For each detected text
    for point in box_list:
        point = point.astype(int)
        x, y, w, h = cv2.boundingRect(point)

        # 计算矩形的对角顶点
        pt1 = (x, y)
        pt2 = (x + w, y + h)

        img_temp = cv2.rectangle(img_temp, pt1, pt2, (255, 255, 255), -1)

        # Convert temp image to black and white for mask
        mask = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

        # "Content-Fill" using mask (INPAINT_NS vs INPAINT_TELEA)
        img_inpaint = cv2.inpaint(img_inpaint, mask, 3, cv2.INPAINT_TELEA)
    input_file = str(uuid.uuid4()) + '.jpg'
    cv2.imwrite(input_file, img_inpaint)
    # resultUrl = upload_file_to_tos(input_file)
    # images.append(resultUrl)
    # if os.path.exists(input_file):
    #     os.remove(input_file)
    # print("0--------")
    return images


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class SingletonType(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonType, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(obj, *args, **kwargs)
        return obj


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)

        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


class DBNET(metaclass=SingletonType):
    def __init__(self, MODEL_PATH):
        self.sess = rt.InferenceSession(MODEL_PATH)

        self.decode_handel = SegDetectorRepresenter()

    def process(self, img, short_size):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h < w:
            scale_h = short_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w

        else:
            scale_w = short_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h

        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)

        img = img.astype(np.float32)

        img /= 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(["out1"], {"input0": transformed_image.astype(np.float32)})
        box_list, score_list = self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        return box_list, score_list


text_handle = DBNET(MODEL_PATH="../models/dbnet.onnx")

