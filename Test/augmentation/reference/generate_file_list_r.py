import os
names = sorted(os.listdir('/dev/shm/reference_images_exp/'))
names = [i[:-4] for i in names]

f = open('dev_reference_exp','a')
for i in range(len(names)):
    if(i==0):
        f.write(names[i]) 
    else:
        f.write('\n' + names[i]) 
f.close()
