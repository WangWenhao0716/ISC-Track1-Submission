# The steps to perform detection

## Test

Download the trained model from [**here**](https://drive.google.com/file/d/1Y-yVSnTfpVB2P1RCfTRAiHA67CTF_ZX9/view?usp=sharing) and save it in the current folder like ```yolov5/best.pt```.

Perform detection by (The images are saved in /dev/shm)
```
CUDA_VISIBLE_DEVICES=0 python detect.py --source /dev/shm/query_images/ --weights best.pt --conf 0.1 
```

Generate the augmented images by
```
python generate.py
```


## Train
If you also want to reproduce the training process of Yolo-V5, please according to the following instructions.

Download the zipped images and labels from [**here**](), unzip it to ```datasets```, and you will get ```yolov5/datasets/isc_v3```.

Download the pre-trained weights from [**here**](https://drive.google.com/file/d/1oZv51z2i8pDlhHqSiGh1vKn1BaL0x9Tb/view?usp=sharing), and save it to ```yolov5/weights/yolov5x.pt ```.

Train Yolo-V5 by:
```
CUDA_VISIBLE_DEVICES=0 python train.py --img-size 640 --batch-size 16 \
--epochs 50 --data ./data/isc.yaml --cfg ./models/yolov5x.yaml --weights weights/yolov5x.pt
```
