import os, sys
import glob
import shutil

data_path = '/home/idealabs/data/opensource_dataset/flowers'

sub_classes = [f for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f))]

for sub_class in sub_classes:
    print(sub_class)
    images = glob.glob(os.path.join(data_path, sub_class, '*.jpg'))
    total_imgs = len(images)
    num_train = total_imgs * 8 //  10
    num_val = total_imgs - num_train
    dst_traindata_path = os.path.join('./data/flowers/train', sub_class)
    dst_valdata_path = os.path.join('./data/flowers/val', sub_class)
    if not os.path.exists(dst_traindata_path):
        os.makedirs(dst_traindata_path)
    if not os.path.exists(dst_valdata_path):
        os.makedirs(dst_valdata_path)
    
    for i in range(0, num_train):
        img = images[i]
        img_name = img.split('/')[-1]
        shutil.copy(img, os.path.join(dst_traindata_path, img_name))
    for i in range(num_train, total_imgs):
        img = images[i]
        img_name = img.split('/')[-1]
        shutil.copy(img, os.path.join(dst_valdata_path, img_name))
     
