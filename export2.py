import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
from easydict import EasyDict as edict
import argparse
from models import build_model
device = torch.device("cpu")
print('device:', device)
checkpoint = torch.load("/data/zhangyong/shenli/ocr_dbnet-master/model_best.pth", map_location=device)

config = checkpoint['config']

config['arch']['backbone']['pretrained'] = False

db_model = build_model(config['arch'])

db_model.load_state_dict(checkpoint['state_dict'])

db_model.to(device)

db_model.eval()

input_name = ['input']

output_name = ['output']

input = Variable(torch.randn(1,3, 1280, 1280))
export_onnx_file="3.onnx"
dynamic_axes = {'input':[2,3], 'output':[2,3] }
torch.onnx.export(db_model, input, export_onnx_file,

input_names=input_name,

output_names=output_name,

verbose=True,

opset_version=11,

export_params=True,

keep_initializers_as_inputs=True,

 dynamic_axes=dynamic_axes
)

print('export done')


