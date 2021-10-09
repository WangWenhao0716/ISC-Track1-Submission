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

These 11 folders are also given in the test part. You should download the trained models by yourselves and store them into the according folders seperately.

## Generate datasets
We augment query and reference datasets to match locally. The codes for augmentation are given in ```augmentation``` folder. By the way, to run theses augmentations, it is assumed that all the original query images and generated images are saved in ```/dev/shm```.

### Query
For query, we design three augmentations, i.e. center cropping, selective search, and detection.

Center cropping: We use center cropping to generate 5 images, and the illustrations are as follows.

![The first center cropping](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_1.pdf)

![The second center cropping](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_2.pdf)

Selective search: We perform selective search and NMS to find the interested parts of an image. Note that selective search is very time-consuming, therefore using multi-cores CPUs manually is highly recommended (Sorry for not developing automatically programmes). 

Detection: We use Yolo-V5 to detect overlay images. The related training and test codes and readme are given in Yolo-V5 Folder.

### Reference
For reference, we only design one augmentation, i.e. dividing. The illustrations are as follows.

![The first dividing](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_3.pdf)

![The second dividing](https://github.com/WangWenhao0716/ISC-Track1-Submission/blob/main/Test/aug_4.pdf)


Note that, though we only have one augmentation, each image can generate 1 + (4 + 1) + (9 + 4) = 19 images, and total 19x1,000,000 = 19,000,000 images are generated . Therefore, please prepare enough storage to store the generated images and corresponding features. Please do NOT store low quality images, which may reduce the performance.


## Test

## Ensemble methods
