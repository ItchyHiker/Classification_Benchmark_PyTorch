import os, sys
import glob
import shutil

data_path = '/home/idealabs/data/opensource_dataset'
'''
traindata_path = os.path.join(data_path, 'tiny-imagenet-200/train')

sub_classes = [f for f in os.listdir(traindata_path) 
        if os.path.isdir(os.path.join(traindata_path, f))]
for sub_class in sub_classes:
    dst_traindata_path = os.path.join('./data/train/', sub_class)
    if not os.path.exists(dst_traindata_path):
        os.makedirs(dst_traindata_path)
    images = glob.glob(os.path.join(traindata_path, sub_class, 'images/*.JPEG'))
    for img in images:
        img_name = img.split('/')[-1]
        shutil.copy(img, os.path.join(dst_traindata_path, img_name))
'''

valdata_path = os.path.join(data_path, 'tiny-imagenet-200/val')
valanno_path = os.path.join(valdata_path,  'val_annotations.txt')

with open(valanno_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split('\t')
        img_name = items[0]
        sub_class = items[1]
        dst_valdata_path = os.path.join('./data/val', sub_class)
        if not os.path.exists(dst_valdata_path):
            os.makedirs(dst_valdata_path)
        shutil.copy(os.path.join(valdata_path, 'images', img_name),
                os.path.join(dst_valdata_path, img_name))
