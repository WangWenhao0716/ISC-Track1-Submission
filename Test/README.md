# The steps for reproducing testing

## Trained models
We first provide 33 trained models to facilitate reproducing. The 33 trained models are obtained according to the provided training codes in training parts by ourselves. The 33 trained models are stored in 11 folders seperately, and you can download them from [**here**](https://drive.google.com/drive/folders/1xVIJdwCDbxVTjmlzo7YXUmlc8JwTUUgA?usp=sharing).

There is a one-to-one relation between the name of training datasets and the name of folders that stores trained models. 

| Training datasets  | Folders storing models |
| ------------- | ------------- |
| isc_100k_256_big_bw  | V5_baseline_BW  |
| isc_100k_256_big | V5_baseline_CC  |
| isc_100k_256_big_blur_bw  | V5_blur_BW  |
| isc_100k_256_big_blur  | V5_blur_CC  |
| isc_100k_256_big_color_p4_bw  | V5_color_BW  |
| isc_100k_256_big_color_p4 | V5_color_CC  |
| isc_100k_256_big_dark | V5_dark_CC  |
| isc_100k_256_big_ff_bw | V5_face_BW  |
| isc_100k_256_big_ff| V5_face_CC  |
| isc_100k_256_big_opa | V5_opa_CC  |
| isc_100k_256_big_u | V5_u_CC  |

These 11 folders are also given in the ```final``` folder. You should download the trained models by yourselves and store them into the according folders seperately.

## Generate datasets
We augment query and reference datasets to match locally. The codes for augmentation are given in ```augmentation``` folder. By the way, to run theses augmentations, it is assumed that all the original query images and generated images are saved in ```/dev/shm```.

### Query
For query, we design four augmentations, i.e. rotating, center cropping, selective search, and detection.

Please remember the original image!

Rotating: We rotate an image 90, 180, and 270 degrees to generate three images.

```
bash rotate.sh
```

Center cropping: We use center cropping to generate 5 images, and the illustrations are as follows.

![The first center cropping](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_1.pdf)

![The second center cropping](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_2.pdf)

```
bash center.sh
```

Selective search: We perform selective search and NMS to find the interested parts of an image. Note that selective search is very time-consuming, therefore using multi-cores CPUs manually is highly recommended. Assume that you have a server with more than 20 CPU cores.

```
bash selective_search_nms.sh
```

Detection: We use Yolo-V5 to detect overlay images. The related training and test codes and readme are given in ```augmentation/query/yolov5``` Folder.

Finally, after all the images are generated, you should generate file list ```dev_queries_exp_VD``` by running 
```
python generate_file_list_q.py
```


For example, if the name of the original image is ```Q00000.jpg```, then we will have the following names after the augmentations: 

Original: ```Q00000_0.jpg```

Rotating: ```Q00000_1.jpg~Q00000_3.jpg```

Center cropping: ```Q00000_4.jpg~Q00000_8.jpg```

Selective search: ```Q00000_10.jpg~Q00000_999.jpg```

Detection: ```Q00000_1000.jpg~...```


### Reference
For reference, we only design one augmentation, i.e. dividing. The illustrations are as follows.

![The first dividing](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_3.pdf)

![The second dividing](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_4.pdf)


Note that, though we only have one augmentation, each image can generate 1 + (4 + 1) + (9 + 4) = 19 images, and total 19x1,000,000 = 19,000,000 images are generated . Therefore, please prepare enough storage to store the generated images and corresponding features. Please do NOT store low quality images, which may reduce the performance.

```
bash dividing.sh
```

Finally, after all the images are generated, you should generate file list ```dev_reference_exp``` by running 
```
python generate_file_list_r.py
```

## Test
Until now, we have 3x11 = 33 trained models, augmented query images, and augmented reference images. The test processing can be divided into two parts: Using augmented query images with original reference images (AQ+OR), and using original query images with augmented reference image (OQ+AR). For AQ+OR, we use all the 33 models, and for OQ+AR, we only use three models. All of the related files are stored in ```final``` folder.

### AQ + OR
All the 33 trained models are used to test. We take V5_baseline_CC (isc_100k_256_big) for instance, the folder has three trained models, i.e. ```baseline_cc_50.pth.tar```, ```baseline_cc_152.pth.tar```, and ```baseline_cc_ibn.pth.tar```. Other 10 folders follow the same pipelines.

We can enter into the folder by ```cd final/V5_baseline_CC```. You should move the generated ```dev_queries_exp_VD``` to ```final/V5_baseline_CC/list_files/```

It is assumed that training images are saved in ```/dev/shm/training_images```, original reference images are saved in ```/dev/shm/reference_images```, and augmented query images are saved in ```/dev/shm/query_images_exp_VD```.

You can gain PCA files and the features of reference images by:
```
bash extract_reference_152.sh
bash extract_reference_50.sh
bash extract_reference_ibn.sh
```
There is no specified sequence for running the above three scripts. However, you should accomplished the three files before running the followings.
Then, extract features of training and augmented query images by:
```
bash extract_training_152.sh
bash extract_training_50.sh
bash extract_training_ibn.sh
bash extract_query_152_VD.sh
bash extract_query_50_VD.sh
bash extract_query_ibn_VD.sh
```

After all the features are extracted, you should run 
```
bash score_normalization_152.sh
bash score_normalization_50.sh
bash score_normalization_ibn.sh
```
***Note that the number of augmented query images is unknown now, we estimate this number, and assume all the features of augmented query images are stored in ```query_{0..28}_byol_VD_50k.hdf5```. However, the number of 28 may be bigger or smaller.***

Finally, we will get 
```
V5_baseline_CC/152/predictions_dev_queries_50k_normalized_exp_VD.csv
V5_baseline_CC/50/predictions_dev_queries_50k_normalized_exp_VD.csv
V5_baseline_CC/ibn/predictions_dev_queries_50k_normalized_exp_VD.csv
```

For all other 10 folders, please perform the same action to get the final ```csv``` files, and all the ```.sh``` files for each folder have been prepared for you.

In conclusion, we will get 33 different ```predictions_dev_queries_50k_normalized_exp_VD.csv``` files in this step.

### OQ + AR

We only use three trained models, i.e. baseline_cc_50.pth.tar, baseline_cc_152.pth.tar, baseline_cc_ibn.pth.tar, to perform testing here.

We can enter into the folder by ```cd final/V5_baseline_CC_ref```. You should move the generated ```dev_reference_exp``` to ```final/V5_baseline_CC_ref/list_files/```.

It is assumed that training images are saved in ```/dev/shm/training_images```, augmented reference images are saved in ```/dev/shm/reference_images_exp```, original query images are saved in ```/dev/shm/query_images```.

You should gain PCA file and features of reference images by:
```
bash extract_reference_exp_152.sh
bash extract_reference_exp_50.sh
bash extract_reference_exp_ibn.sh
```
There is no specified squence for running the three scripts. However, you should accomplished the three files before running the followings. Then, extract features of training and original query images by:
```
bash extract_training_152.sh
bash extract_training_50.sh
bash extract_training_ibn.sh
bash extract_query_152.sh
bash extract_query_50.sh
bash extract_query_ibn.sh
```
After all the features are extracted, you should run
```
bash score_normalization_152.sh
bash score_normalization_50.sh
bash score_normalization_ibn.sh
```
Finally, we will get
```
V5_baseline_CC_ref/152/predictions_dev_queries_50k_normalized_exp.csv
V5_baseline_CC_ref/50/predictions_dev_queries_50k_normalized_exp.csv
V5_baseline_CC_ref/ibn/predictions_dev_queries_50k_normalized_exp.csv
```
In conclusion, we will get 3 different ```predictions_dev_queries_50k_normalized_exp.csv``` files in this step.

## Ensemble methods

***Note that though ensemble methods are performed to all images together, it can be also apply to the circumstance that when there are only a query and a reference image by adjusting the threshold.***

### How to gather the 3 results getting from 152, 50, ibn models (AQ + OR).
We still take ```V5_baseline_CC``` for example, and other 10 folders follow similar pipelines.

First enter the folder, by ```cd final/V5_baseline_CC```, then run ```python aggregate.py```. You will get ```V5-baseline-CC-234-50k-VD.csv``` file in the ```V5_baseline_CC``` folder.

### How to gather the 3 results getting from 152, 50, ibn models (OQ + AR).

First enter the folder, by ```cd final/V5_baseline_CC_ref```, then run ```python aggregate.py```. You will get ```R-baseline-CC-234-50k.csv``` file in the ```V5_baseline_CC_ref``` folder.

### How to gather the 12 results from 12 folders (AQ+OR & OQ+AR).

Before gathering, we should generate 4 files:

1~3: The augmented query images that smaller than 100, 150, 200:
```
python generate_100_150_200.py
```
You will get ```query_wrong_100_VD.npy```, ```query_wrong_150_VD.npy```, and ```query_wrong_100_VD.npy``` in ```final``` folder. 

4: The augmented reference images that generated from face images.
```
python detect_face.py
```
You will get ```face_del.npy``` in ```final``` folder (*In fact, we have prepared the file for you, and there is no need to generate it again.*). 

Finally, by running ```python combine.py```, you will get ```step_11_50k.csv``` in ```final``` folder.

### How to perform multi-scale ensemble.
The image size used in the ```final``` is 256. We should repeat all the above pipelines using image size of 200 and 320 in ```final_200``` and ```final_320``` folders, respectively. To be efficient, OQ+AR should NOT be repeated, you can copy the ```R-baseline-CC-234-50k.csv``` from ```final/V5_baseline_CC_ref``` to ```final_200/V5_baseline_CC_ref``` and ```final_320/V5_baseline_CC_ref``` directly.

### How to get the final submission.
Until now, we have ```final/step_11_50k.csv```, ```final_200/step_11_50k.csv```, and ```final_320/step_11_50k.csv```. 
By running ```bash submit.sh```, you can get the file, ```submit_50k.csv```, to submit.

Congratulations!

## Another choice
Remember that we have got three trained models, i.e. ```baseline_cc_50_pair.pth.tar```, ```baseline_cc_152_pair.pth.tar```, and ```baseline_cc_ibn_pair.pth.tar```. You can just copy ```V5_baseline_CC``` folder by ```cp -r V5_baseline_CC V5_baseline_CC_pair```, and replace ```xxx.pth.tar``` with ```xxx_pair.pth.tar```.
Then perform testing and ensembling same as in ```V5_baseline_CC```. You should only do this in ```final``` folder. 

Finally, you can get ```V5_baseline_CC_pair/V5-baseline-CC-234-50k-VD-pair.csv```.

Now, we have ```submit_50k.csv``` and ```V5-baseline-CC-234-50k-VD-pair.csv```. By running
```
python aggregate_choice2.py
```
You can get the second file, ```submit_50k_choice2.csv```, to submit.





