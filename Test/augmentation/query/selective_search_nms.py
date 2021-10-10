from __future__ import (
    division,
    print_function,
)

import os
from PIL import Image
import skimage.data
import selectivesearch
import numpy as np

def NMS(arr, thresh):
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    score = np.array([1]*len(arr))#arr[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        inter = w * h
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        index = np.where(ious <= thresh)[0]
        order = order[index+1]

    return keep

path = '/dev/shm/query_images/'
names = sorted(os.listdir(path))
# To perform multi-core running manually, you should change the num, begin, end to seperate images into multi-parts, for example: use one core to deal with 2500 images .

#num = 19
begin = 0 #num * 2500
end = 50000 #(num+1) * 2500
for i in range(begin,end):
    print("processing ... %d"%i)
    test = Image.open(path + names[i])
    img = np.array(test)
    
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 2 or h / w > 2:
            continue
        candidates.add(r['rect'])
    if(len(candidates)==0):
        continue
    ar = np.array(list(candidates))
    ar[:,2] = ar[:,2] + ar[:,0]
    ar[:,3] = ar[:,3] + ar[:,1]
    keep = NMS(ar, thresh=0.5)
    candidates = ar[keep]
    print("The length of the proposals is %d"%len(candidates))
    for num in range(len(candidates)):
        twt = test.crop(candidates[num])
        num_fix = 10 + num
        twt.save("/dev/shm/query_images_exp/" + names[i][:-4] + '_' + str(num_fix) + ".jpg", quality=100)

#10~
