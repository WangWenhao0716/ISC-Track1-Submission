# ISC-Track1-Submission
The codes and related files to reproduce the results for Image Similarity Challenge Track 1.

## Required dependencies
To begin with, you should install the following packages with specified version in Python. Other versions may work but please do NOT try. For instance, cuda 11.0 has some bugs which bring very bad results. The hardware chosen is Nvidia Tesla V100. Other hardwares, such as A100, may work but please do NOT try. The stability is not garanteed, for instance, the Ampere architecture is not suitable and some instability is observed.

* Python 3.7.10
* Pytorch 1.7.1 with cuda 10.1
* faiss-gpu 1.7.1 with cuda 10.1
* h5py 3.4.0
* pandas 1.3.3
* sklearn 1.0
* PIL 8.3.2
* Numpy 1.16.0
* Torchvision 0.8.2 with cuda 10.1

## Pre-trained models
We use three pre-trained models. They are all pre-trained on ImageNet unsupervisedly. To be convenient, we first directly give the pretrained models as follows, then also the training codes are given.

The first backbone: [**ResNet-50**](https://drive.google.com/file/d/14M57frgk3TX-yLF8diwALLHtPdCZ53mS/view?usp=sharing); The second backbone: [**ResNet-152**](https://drive.google.com/file/d/1-1QkeKCo9PrgDdUF3fe561JtEntd32hv/view?usp=sharing); The third backbone: [**ResNet-50-IBN**](https://drive.google.com/file/d/1-5B2B5VherIRHN9ahE-5L6w1VoWxBD_c/view?usp=sharing).


## Training


## Test
