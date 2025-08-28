import onnxruntime as rt
import numpy as np
import uuid
import cv2
import torch

from .decode import SegDetectorRepresenter
from typing import Any, Dict, List, Literal, Optional, Union
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)



def textRemove(oriImg):
    print("0--------")
    img = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

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
    img_np = img_inpaint
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)

    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, C)
    # resultUrl = upload_file_to_tos(input_file)
    # images.append(resultUrl)
    # if os.path.exists(input_file):
    #     os.remove(input_file)
    # print("0--------")
    return img_tensor


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
        sess_opts = rt.SessionOptions()
        sess_opts.inter_op_num_threads = 8  # 设置操作间并行线程数
        sess_opts.intra_op_num_threads = 8  # 设置操作内部并行线程数

        self.sess = rt.InferenceSession(MODEL_PATH, sess_opts)

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
        print("scale:", scale_h, scale_w)
        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
        print("astype:", scale_h, scale_w)
        img = img.astype(np.float32)

        img /= 255.0
        img -= mean
        img /= std
        print("transpose:", scale_h, scale_w)

        img = img.transpose(2, 0, 1)
        print("expand_dims:", scale_h, scale_w)

        transformed_image = np.expand_dims(img, axis=0)
        print("run:", scale_h, scale_w)
        out = self.sess.run(["out1"], {"input0": transformed_image.astype(np.float32)})
        box_list, score_list = self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        print("box_list:", scale_h, scale_w)    
        return box_list, score_list


model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dbnet.onnx")
text_handle = DBNET(MODEL_PATH=model_path)

# 测试textRemove
if __name__ == "__main__":
    import time

    img_path = "img_1.png"
    img = cv2.imread(img_path)
    start = time.time()
    images = textRemove(img)
    end = time.time()
    print("time:", end - start)
