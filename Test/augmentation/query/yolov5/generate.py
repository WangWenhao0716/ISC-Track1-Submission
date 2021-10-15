import os
from PIL import Image
import numpy as np

names = ['Q%05d'%i for i in range(50000)]
path = '/dev/shm/query_images/'


al = []
for i in range(len(names)):
    if os.path.exists(path+names[i]+'.jpg.npy'):
        detect = np.load(path+names[i]+'.jpg.npy')
        for j in range(len(detect)):
            class_ = detect[j][-1]
            if (class_==0):
                score = detect[j][-2]
                if (score>0.01):
                    old_img = Image.open(path+names[i]+'.jpg')
                    w, h = old_img.size
                    max_length = max(w,h)
                    enlarge = 640/max_length
                    new_w = int(enlarge*w)
                    new_h = int(enlarge*h)
                    old_img = old_img.resize((new_w,new_h))
                    new_img = old_img.crop(detect[j][:-2])
                    num = 1000 + j
                    dst = '/dev/shm/query_images_exp_VD/' + names[i] + '_' + str(num) + '.jpg'
                    print(i)
                    al.append(i)
                    new_img.save(dst,quality=100)
#from 100
