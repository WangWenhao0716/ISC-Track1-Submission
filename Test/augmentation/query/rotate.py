import os
from PIL import Image
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

os.makedirs('/dev/shm/query_images_exp/', exist_ok = True)
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
