from PIL import Image
from PIL import ImageFilter
import numpy as np
import os
import argparse

names = sorted(os.listdir('/dev/shm/query_images/'))
assert len(names)==50000

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * 2500
end = (num+1) * 2500

for n in range(begin,end):
    if(n%100==0):
        print(n)
    name = names[n]
    im = Image.open("/dev/shm/query_images/"+name)
    size = im.size
    
    
    im.save('/dev/shm/query_images_exp/' + name[:-4]+'_0' +'.jpg',quality=100)
    
    #center
    num = 4
    img_c = im.crop(box=(int(size[0]*0.25),int(size[1]*0.25),int(size[0]*0.75),int(size[1]*0.75)))
    img_c.save('/dev/shm/query_images_exp/' + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    #center-4
    num = num + 1
    img_c = im.crop(box=(int(size[0]*1/6),int(size[1]*1/6),int(size[0]*3/6),int(size[1]*3/6)))
    img_c.save('/dev/shm/query_images_exp/' + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    num = num + 1
    img_c = im.crop(box=(int(size[0]*3/6),int(size[1]*1/6),int(size[0]*5/6),int(size[1]*3/6)))
    img_c.save('/dev/shm/query_images_exp/' + name[:-4]+'_' + str(num)+'.jpg',quality=100)

    num = num + 1
    img_c = im.crop(box=(int(size[0]*1/6),int(size[1]*3/6),int(size[0]*3/6),int(size[1]*5/6)))
    img_c.save('/dev/shm/query_images_exp/' + name[:-4]+'_' + str(num)+'.jpg',quality=100)

    num = num + 1
    img_c = im.crop(box=(int(size[0]*3/6),int(size[1]*3/6),int(size[0]*5/6),int(size[1]*5/6)))
    img_c.save('/dev/shm/query_images_exp/' + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
# rotate aug: _4, _5, _6, _7, _8 (ori: _0)
