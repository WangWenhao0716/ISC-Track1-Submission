from PIL import Image
from PIL import ImageFilter
import numpy as np
import os

names = sorted(os.listdir('/dev/shm/reference_images/'))
assert len(names)==1000000
#num = 9
begin = 0#num*5000
end = 1000000#(num+1)*5000
for n in range(begin,end):
    if(n%100==0):
        print(n)
    name = names[n]
    im = Image.open("/dev/shm/reference_images/"+name)
    path_result = '/dev/shm/reference_images_exp/'
    
    size = im.size
    
    xs = []
    xs.append(0)
    xs.append(size[0]//2)
    xs.append(size[0])
    ys = []
    ys.append(0)
    ys.append(size[1]//2)
    ys.append(size[1])
    
    num = 0
    im.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    if((len(xs)-1)*(len(ys)-1) != 1):
        for i in range(len(ys)-1):
            for j in range(len(xs)-1):
                num = num + 1
                img_c = im.crop(box=(xs[j], ys[i], xs[j+1], ys[i+1]))
                img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    num = num + 1
    img_c = im.crop(box=(int(size[0]*0.25),int(size[1]*0.25),int(size[0]*0.75),int(size[1]*0.75)))
    img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    xs = []
    xs.append(0)
    xs.append(int(size[0]*1/3))
    xs.append(int(size[0]*2/3))
    xs.append(size[0])
    ys = []
    ys.append(0)
    ys.append(int(size[1]*1/3))
    ys.append(int(size[1]*2/3))
    ys.append(size[1])
    if((len(xs)-1)*(len(ys)-1) != 1):
        for i in range(len(ys)-1):
            for j in range(len(xs)-1):
                num = num + 1
                img_c = im.crop(box=(xs[j], ys[i], xs[j+1], ys[i+1]))
                img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    num = num + 1
    img_c = im.crop(box=(int(size[0]*1/6),int(size[1]*1/6),int(size[0]*3/6),int(size[1]*3/6)))
    img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
    
    num = num + 1
    img_c = im.crop(box=(int(size[0]*3/6),int(size[1]*1/6),int(size[0]*5/6),int(size[1]*3/6)))
    img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)

    num = num + 1
    img_c = im.crop(box=(int(size[0]*1/6),int(size[1]*3/6),int(size[0]*3/6),int(size[1]*5/6)))
    img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)

    num = num + 1
    img_c = im.crop(box=(int(size[0]*3/6),int(size[1]*3/6),int(size[0]*5/6),int(size[1]*5/6)))
    img_c.save(path_result + name[:-4]+'_' + str(num)+'.jpg',quality=100)
