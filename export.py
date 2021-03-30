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

input = Variable(torch.randn(1,3,640,640))
y=db_model(input)
export_onnx_file="dynamic.onnx"
torch.onnx.export(db_model, input, export_onnx_file,

input_names=input_name,

output_names=output_name,

verbose=False,
do_constant_folding=True,
opset_version=12,
training=False,
export_params=True,

#keep_initializers_as_inputs=True,

dynamic_axes={"input": {2: "height",3:"width"}},

# "output": {3: "time_step"}}

)

print('export done')
