# The steps to get the pre-trained models

## ResNet-50

As declared before, you can directly download the pre-trained models from the [**link**](https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth) given by the project [**Barlow Twins**](https://github.com/facebookresearch/barlowtwins). We do not re-pre-train the model.


## ResNet-152 and ResNet-50-IBN

We modifiy the official codes of [**Momentum2-teacher**](https://github.com/zengarden/momentum2-teacher) by changing the backbones to ResNet-152 and ResNet-50-IBN. The folder momentum2-teacher-resnet152 and momentum2-teacher-resnetIBN give the details.

It is assumed that the ImageNet dataset is saved in the shared memory, and the data structure is:
```
/dev/shm
      *ILSVRC2012_RAW_PYTORCH
         *train
             *n02443484
             *n02701002
         *val
             *n02443484
             *n02701002
```

### ResNet-152
Please enter the folder by ```cd momentum2-teacher-resnet152```, and running:
```
python train.py -b 512 -d 0-7 \
--experiment-name imagenet_baseline_resnet152 \
-f momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_300e_exp.py \
```
on a standard 8 V100 GPUs machine.
You will get the ```last_epoch_ckpt.pth.tar``` after 300 epochs in the path ```./outputs/imagenet_baseline_resnet152```.

After getting the pretrained model, run:
```
python tran.py
```
to transfer the unsupervised pretrained model to the one we can use. It is saved in ```/dev/shm``` by default.


### ResNet-50-IBN
Please enter the folder by ```cd momentum2-teacher-resnetIBN```, and running:
```
python train.py -b 1024 -d 0-7 \
--experiment-name imagenet_baseline_resnet50ibn \
-f momentum_teacher/exps/arxiv/exp_8_v100/momentum2_teacher_300e_exp.py \
```
on a standard 8 V100 GPUs machine.
You will get the ```last_epoch_ckpt.pth.tar``` after 300 epochs in the path ```./outputs/imagenet_baseline_resnet50ibn```.

After getting the pretrained model, run:
```
python tran.py
```
to transfer the unsupervised pretrained model to the one we can use. It is saved in ```/dev/shm``` by default.


Note: The above pre-training codes support training from a checkpoint.







