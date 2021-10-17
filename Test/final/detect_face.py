import face_recognition, os
from PIL import Image

path = '/dev/shm/reference_images/'
path_new = '/dev/shm/reference_images_exp_face/'

os.makedirs(path_new, exist_ok=True)

ls = sorted(os.listdir(path))
assert len(ls) == 100_0000
faces = []

for i in range(0,1000000):
    image = face_recognition.load_image_file(path+ls[i])
    face_locations = face_recognition.face_locations(image, model="cnn")
    if(len(face_locations)!=0):
        print(ls[i])
        faces.append(ls[i])
        img = Image.open(path+ls[i])
        face_locations_c = []
        for k in range(len(face_locations)):
            p_0 = face_locations[k][3]-img.size[0]//8
            p_1 = face_locations[k][0]-img.size[1]//8
            p_2 = face_locations[k][1]+img.size[0]//8
            p_3 = face_locations[k][2]+img.size[1]//8
            face_locations_c.append((p_0,p_1,p_2,p_3))
        for j in range(len(face_locations_c)):
            img_crop = img.crop(face_locations_c[j])
            name = ls[i].split('.')[0] + '_' + str(20+j) + '.jpg'
            img_crop.save(path_new + name, quality=100)

            
imgs = sorted(os.listdir(path_new))
import numpy as np
fff = np.array(sorted(list(set([i.split('_')[0] for i in imgs]))))
np.save('face_del.npy',fff)
