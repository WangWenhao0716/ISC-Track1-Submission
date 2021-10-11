import os
names = sorted(os.listdir('/dev/shm/query_images_exp_VD/'))
names = [i[:-4] for i in names]

f = open('dev_queries_exp_VD','a')
for i in range(len(names)):
    if(i==0):
        f.write(names[i]) 
    else:
        f.write('\n' + names[i]) 
f.close()
