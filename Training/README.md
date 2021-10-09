# The steps for reproducing training

## Generate datasets

In the training parts, we use 11 datasets with different augmentations. The datasets are all selected from the provided training data. We choose 100,000 out of 1000,0000 images to perform training. It should be noted that NO reference data is used.

To be convenient, we supply the link to generated 11 datasets as follows.

* [**isc_100k_256_big_bw**](test)
* [**isc_100k_256_big**](test) 
* [**isc_100k_256_big_blur_bw**]()
* [**isc_100k_256_big_blur**]()
* [**isc_100k_256_big_color_p4_bw**]()
* [**isc_100k_256_big_color_p4**]()
* [**isc_100k_256_big_dark**]()
* [**isc_100k_256_big_ff_bw**]()
* [**isc_100k_256_big_ff**]()
* [**isc_100k_256_big_opa**]()
* [**isc_100k_256_big_u**]()


You can directly download them from Google drive and unzip them. The default path is ```/dev/shm``` to store the images temporarily for training.


Or you can generate the training datasets according to the codes in the ```generate``` folder by yourself. It takes about about one day to generate one dataset using one core of Intel Golden 6240 CPU. To speed up, using multi-cores is a feasible way. We use some images from [**OpenImage**](https://opensource.google/projects/open-images-dataset) to generate overlay and underlay augmentation under CC-by 4.0 License. It should be noted that the using of OpenImage is not a must, other images show similar performance. The part of OpenImage we used can be downloaded from [**here**](). 

Assuming all the datasets are stored in ```/dev/shm```. An example to generate a dataset is:
```
cd generate && python isc_100k_256_big.py
```
The process to generate other 8 datasets (except for ```isc_100k_256_big_ff``` and ```isc_100k_256_big_ff_bw```) is similiar with this command line.

For ```isc_100k_256_big_ff```, we select some images which contain human faces from the given training dataset and perform the same augmentation as ```isc_100k_256_big```. Finally, the augmented face images are added into the ```isc_100k_256_big``` dataset to form ```isc_100k_256_big_ff```. And ```isc_100k_256_big_ff_bw``` is the black and white version of ```isc_100k_256_big_ff```.


## Training

Remember that we have three pre-trained models, i.e. ```resnet50_bar.pth```, ```unsupervised_pretrained_byol_152.pkl```, and ```unsupervised_pretrained_m2t_ibn.pkl```, stored in ```/dev/shm```. For one dataset, we train 3 models according to the three pre-trained models. Therefore, totally, 3x11=33 models are trained. To be convenient, we give all the 33 training scripts in ```scirpt``` folder. When using one script, you should move it by ```mv xxx.sh ../```. 

Take the first script, ```Train_baseline_CC_50.sh```, for instance. You should
```
mv Train_baseline_CC_50.sh ../
```
```
bash Train_baseline_CC_50.sh
```

Take a look to the ```Train_baseline_CC_50.sh```:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_high_balance.py \
-ds isc_100k_256_big -a resnet50 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /dev/shm/ \
--logs-dir logs/baseline_CC/50 \
--height 256 --width 256
```
The ```/dev/shm``` is the dir to store images, for ```isc_100k_256_big``` dataset, please check the number of images is 2,000,000. The checkpoints will be saved into ```logs/baseline_CC/50```. And the final checkpoint, i.e. ```checkpoint_24.pth.tar``` will be used to test. Please do NOT change any hyper-parameters in any scripts. Also, to be efficient, you should use the ```Tran.py``` to discard all the fully-connected layers. 

### One more thing
Due to the large number of classes, we split the fully-connected layers into 4 GPUs, therefore, if the number of images cannot be divided by 4, an error will occur. This error may happen for ```isc_100k_256_big_ff_bw``` and ```isc_100k_256_big_ff``` datasets, and to eliminate the error, you can go to the folder by ``` cd /dev/shm/isc_100k_256_big_ff_bw/isc_100k_256_big_ff_bw``` and delete some images, such as ```rm -rf 0_*```. You should NOT delete more than 3 IDs.







