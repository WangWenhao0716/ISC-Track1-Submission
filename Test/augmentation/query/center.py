from PIL import Image
from PIL import ImageFilter
import numpy as np
import os

names = sorted(os.listdir('/dev/shm/query_images/'))
assert len(names)==50000
#num = 9
begin = 0 #num*5000
end = 50000 #(num+1)*5000
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
