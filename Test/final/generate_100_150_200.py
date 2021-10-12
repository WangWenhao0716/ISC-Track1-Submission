from PIL import Image
import os
import numpy as np

path = '/dev/shm/query_images_exp_VD/'
names = sorted(os.listdir(path))

query_wrong_100 = []
for i in range(len(names)):
    img = Image.open(path+names[i])
    size = img.size
    if size[0]<100 or size[1]<100:
        query_wrong_100.append(names[i])
    if(i%10000==0):
        print(i)
query_wrong_100 = [im[:-4] for im in query_wrong_100]
np.save("./query_wrong_100_VD.npy",np.array((query_wrong_100)))

query_wrong_150 = []
for i in range(len(names)):
    img = Image.open(path+names[i])
    size = img.size
    if size[0]<150 or size[1]<150:
        query_wrong_150.append(names[i])
    if(i%10000==0):
        print(i)
query_wrong_150 = [im[:-4] for im in query_wrong_150]
np.save("./query_wrong_150_VD.npy",np.array((query_wrong_150)))

query_wrong_200 = []
for i in range(len(names)):
    img = Image.open(path+names[i])
    size = img.size
    if size[0]<200 or size[1]<200:
        query_wrong_200.append(names[i])
    if(i%10000==0):
        print(i)
query_wrong_200 = [im[:-4] for im in query_wrong_200]
np.save("./query_wrong_200_VD.npy",np.array((query_wrong_200)))
