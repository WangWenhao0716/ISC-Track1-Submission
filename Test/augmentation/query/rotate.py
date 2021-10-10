import os
from PIL import Image

names = sorted(os.listdir('/dev/shm/query_images/'))
assert len(names)==50000
#num = 9
begin = 0 #num*5000
end = 50000 #(num+1)*5000
for i in range(begin,end):
    if(i%100==0):
        print(i)
    img = Image.open('/dev/shm/query_images/' + names[i])
    angles = [Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
    num = 1
    for angle in angles:
        rotated = img.transpose(angle)
        name_new = '/dev/shm/query_images_exp/' + names[i][:-4] + '_' + str(num) + '.jpg'
        rotated.save(name_new,quality=100)
        num = num + 1

        
# rotate aug: _1, _2, _3
