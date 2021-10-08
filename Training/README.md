# The steps for reproducing training

## Generate datasets.

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
The process to generate other 10 datasets is similiar with this command line.

