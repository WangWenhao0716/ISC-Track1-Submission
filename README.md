
# ISC-Track1-Submission (Rank 1)
The codes and related files to reproduce the results for Image Similarity Challenge Track 1.

# News
2022.4.14 Updates: The license is changed to CC BY-NC 4.0. No one is permitted to use the current version code for commerical without the permission from Wenhao Wang. Thanks for understanding!

2021.11.25 Updates: This solution is verified! If you find this code useful for your research, please cite our paper.


2021.11.24 Updates: Fix some bugs without changing performance.

## Required dependencies
To begin with, you should install the following packages with the specified versions in Python, Anaconda. Please do not use cuda 11.0, which has some bugs. The hardware chosen is Nvidia Tesla V100 and Intel CPU. We also reproduce the experiments using DGX A100 with AMD CPU, with pytorch 1.9.1 and cuda 11.1.

* python 3.7.10
* pytorch 1.7.1 with cuda 10.1
* faiss-gpu 1.7.1 with cuda 10.1
* h5py 3.4.0
* pandas 1.3.3
* sklearn 1.0
* skimage 0.18.3
* PIL 8.3.2
* cv2 4.5.3.56
* numpy 1.16.0
* torchvision 0.8.2 with cuda 10.1
* augly 0.1.4
* selectivesearch 0.4
* face-recognition 1.3.0 (with dlib of gpu-version)
* tqdm 4.62.3
* requests 2.26.0
* seaborn 0.11.2
* mkl 2.4.0
* loguru 0.5.3

Note: Some unimportant packages may be missing, please install them using pip directly when an error occurs.

## Pre-trained models
We use three pre-trained models. They are all pre-trained on ImageNet unsupervisedly. To be convenient, we first directly give the pre-trained models as follows, then also the training codes are given.

The first backbone: [**ResNet-50**](https://drive.google.com/file/d/14M57frgk3TX-yLF8diwALLHtPdCZ53mS/view?usp=sharing); The second backbone: [**ResNet-152**](https://drive.google.com/file/d/1-1QkeKCo9PrgDdUF3fe561JtEntd32hv/view?usp=sharing); The third backbone: [**ResNet-50-IBN**](https://drive.google.com/file/d/1-5B2B5VherIRHN9ahE-5L6w1VoWxBD_c/view?usp=sharing).

For ResNet-50, we do not pre-train it by ourselves. It is directly downloaded from [**here**](https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth). It is supplied by Facebook Research, and the project is [**Barlow Twins**](https://github.com/facebookresearch/barlowtwins). You should rename it to ```resnet50_bar.pth```.

For ResNet-152 and ResNet-50-IBN, we use the official codes of [**Momentum2-teacher**](https://github.com/zengarden/momentum2-teacher). We only change the backbone to ResNet-152 and ResNet-50-IBN. It takes about 2 weeks to pre-train the ResNet-152, and 1 week to pre-train the ResNet-50-IBN on 8 V100 GPUs. To be convenient, we supply the whole pre-training codes in the ```Pretrain``` folder. The related readme file is also given in that folder. 

It should be noted that pre-training processing plays a very important role in our algorithm. Therefore, if you want to reproduce the pre-trained results, please do NOT change the number of GPUs, the batch size, and other related hyper-parameters.


## Training
For training, we generate 11 datasets. For each dataset, 3 models with different backbones are trained. Each training takes about/less than 1 day on 4 V100 GPUs (bigger backbone takes longer and smaller backbone takes shorter). The whole training codes, including how to generate training datasets and the link to the generated datasets, are given in the ```Training``` folder. For more details, please refer to the readme file in that folder.


## Test
To test the performance of the trained model, we perform multi-scale, multi-model, and multi-part testing and ensemble all the scores to get the final score. To be efficient, 33 V100 GPUs are suggested to use. The time for extracting all query images' features using 33 V100 GPUs is about 3 hours. Also extracting and storing training and reference images' features take a lot of time. Please be patient and prepare enough storage to reproduce the testing process. We give all the information to generate our final results in the ```Test``` folder. Please reproduce the results according to the readme file in that folder.

## Citation
```
@article{wang2021d,
  title={D\^{} 2LV: A Data-Driven and Local-Verification Approach for Image Copy Detection},
  author={Wang, Wenhao and Sun, Yifan and Zhang, Weipu and Yang, Yi},
  journal={arXiv preprint arXiv:2111.07090},
  year={2021}
}
```
