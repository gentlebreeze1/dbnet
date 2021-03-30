# 中文检测DBNet

This repo provides training and demo code of DBnet  text detection on Chinese characters.



## Data Preparation

Training data: prepare a text `train.txt` in the following format, use '\t' as a separator
```
./datasets/train/img/001.jpg	./datasets/train/gt/001.txt
```

Validation data: prepare a text `test.txt` in the following format, use '\t' as a separator
```
./datasets/test/img/001.jpg	./datasets/test/gt/001.txt
```
- Store images in the `img` folder
- Store groundtruth in the `gt` folder

The groundtruth can be `.txt` files, with the following format:
```
x1, y1, x2, y2, x3, y3, x4, y4, annotation
```


## Train
Simply run:

```
./single_gpu_train.sh
```

This will train on ICPR dataset.



## Demo

We have a quick demo to visualize detection result:

```
 python3 demo.py --model_path output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth --data ./imgs/
```

The pretrained model can be download from manaai.cn.

# TENSORRT
## pth2onnx
执行 python export.py既可以导出onnx，根据实际情况决定是否需要dynamic_axes
## onnx2trt
执行python conver_trt_quant.py既可以获取tensorrt的二进制文件
## inference
demo_trt和demo2_trt提供两种解码格式对应decode_v2和decode其中decode_v2更复杂更精确，decode相对简单
动态输入的结果要比固定输入的结果更精确，目前本版本tensorrt不支持动态输入

