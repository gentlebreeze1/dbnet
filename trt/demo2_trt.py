#coding:utf-8
import random
from PIL import Image
import numpy as np
import cv2
import time
import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
from trt.decode import SegDetectorRepresenter
import sys, os
from trt.common import *
size=640
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
def get_engine1(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
class dbnet_code:
    def __init__(self, model_path):
        self.engine=get_engine1(model_path)
        self.context=self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
    def resize_image(self, img, min_scale=size, max_scale=1088):

        height, width, _ = img.shape
        if height < width:
            new_width= min_scale
            new_height = int(min_scale/width*height)
            top,right,left=0,0,0
            bottom=min_scale-new_height
        else:
            new_width = int(min_scale/height*width)
            new_height = min_scale
            left,bottom,top=0,0,0
            right=min_scale-new_width
        re_im = cv2.resize(img, (new_width, new_height))
        re_im=cv2.copyMakeBorder(re_im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(255,255,255))
        return re_im

    def predict(self, img, min_scale=size):
        # with self.engine.create_execution_context() as context:
        img = self.resize_image(img, min_scale=min_scale)
        self.load_normalized_test_case(img, self.inputs[0].host)
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                             stream=self.stream)
        preds = trt_outputs[0].reshape(1, 2, size, size)
        mask = preds[0, 0, ...]
        batch = {'shape': [(size, size)]}
        box_list = SegDetectorRepresenter(thresh=0.5, box_thresh=0.7, max_candidates=1000,
                                                      unclip_ratio=1.5)(batch, preds)
        # box_list, score_list = box_list[0], score_list[0]
        is_output_polygon = False


        return box_list[0]
    def load_normalized_test_case(self, img, pagelocked_buffer):
        # Converts the input image to a CHW Numpy array
        def normalize_image(image):
            # Resize, antialias and transpose the image to CHW.
            # c, h, w = ModelData.INPUT_SHAPE
            # image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS))#.transpose([2, 0, 1])#.astype(trt.nptype(ModelData.DTYPE)).ravel()
            # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
            mean_ = np.array([0.485, 0.456, 0.406])
            std_ = np.array([0.229, 0.224, 0.225])
            img = (image / 255. - mean_) / std_
            return np.asarray(img).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()
        # Normalize the image and copy to pagelocked memory.
        np.copyto(pagelocked_buffer, normalize_image(img))
        return img

def draw_bbox(img, result, color=(0, 0, 255), thickness=2):
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img
def draw_new(img, result, color=(0, 0, 255), thickness=2):
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img
# def draw_new(img,result,color=(0, 0, 255), thickness=2):
#
def debug_main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    onnx_model_file = '/data/zhangyong/shenli/ocr_dbnet-master/trt/model/dbnet.trt'
    model = dbnet_code(onnx_model_file)
    # path = './第七批手机拍摄错误图片'
    # output_path = './第七批手机拍摄错误图片_条形码检测'

    path = '/data/zhangyong/shenli/ocr_dbnet-master/trt/img'
    output_path = '/data/zhangyong/shenli/ocr_dbnet-master/trt/out'
    # 保存结果到路径
    os.makedirs(output_path, exist_ok=True)
    imgs_list_path = [os.path.join(path, i) for i in os.listdir(path)]
    times = []
    for i, img_list_path, in enumerate(imgs_list_path):
    # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # probability that the image corresponds to that label
        print('==img_list_path:', img_list_path)
        img = cv2.imread(img_list_path)
        pred_path = os.path.join(output_path, img_list_path.split('/')[-1].split('.')[0] + '_pred.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st_time = time.time()
        box_list= model.predict(img)
        times.append(time.time() - st_time)

        draw_img = draw_bbox(model.resize_image(img, min_scale=size).copy(), box_list)
        cv2.imwrite(pred_path.replace('pred', 'draw'), draw_img)
    print(times)
    print('平均时间为{}'.format(sum(times) / len(times)))

if __name__ == '__main__':
    debug_main()
