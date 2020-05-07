import os, sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr
from torchsummary import summary

import config
from nets.mobilenet_v2 import MobileNetV2
from tools.dataset import Dataset
from tools.utils import get_lr
from trainer import  Trainer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# load dataset
train_dataset = Dataset('data/{0}/train'.format(config.attribute), is_train=True)
train_loader = train_dataset.load_data()
val_dataset = Dataset('data/{0}/val'.format(config.attribute), is_train=False)
val_loader = val_dataset.load_data()

# create model
model = MobileNetV2(n_class=config.num_classes, width_mult=1.0)
summary(model.cuda(), (3, 224, 224))
model.load_state_dict(torch.load('pretrained_weights/{}_mobilenetv2_epoch_200.pth'.format(config.attribute)), strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lr.MultiStepLR(optimizer, milestones=[20, 80, 160], gamma=0.1)
# lr_scheduler = lr.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=200)  
trainer = Trainer(train_loader, val_loader, model, optimizer, lr_scheduler,
                    config.epochs, device, config.save_path)
trainer.train()
