import os
names = sorted(os.listdir('/dev/shm/'))
names = ['/dev/shm/' + name for name in names if 'reference_images_exp_' in name]
all_names = []
for i in range(len(names)):
    ls = sorted(os.listdir(names[i]))
    for j in range(len(ls)):
        all_names.append(names[i] + '/' + ls[j])

f = open('dev_reference_exp','a')
for i in range(len(all_names)):
    if(i==0):
        f.write(all_names[i]) 
    else:
        f.write('\n' + all_names[i]) 
f.close()
