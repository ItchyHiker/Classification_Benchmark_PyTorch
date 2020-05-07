import os, sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

import config
from nets.mobilenet_v2_relu import MobileNetV2
from tools.dataset import Dataset
from tools.utils import get_lr
from trainer_prune import  Trainer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# load dataset
train_dataset = Dataset('data/{0}/train'.format(config.attribute), is_train=True)
train_loader = train_dataset.load_data()
val_dataset = Dataset('data/{0}/val'.format(config.attribute), is_train=False)
val_loader = val_dataset.load_data()

# create model
model = MobileNetV2(n_class=config.num_classes)
model.load_state_dict(torch.load('results/check_points/{}_mobilenetv2_epoch_200.pth'.format(config.attribute)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = MultiStepLR(optimizer, milestones=[20, 80, 160], gamma=0.1)

trainer = Trainer(train_loader, val_loader, model, optimizer, lr_scheduler,
                    config.epochs, config.prune_epochs, device, config.save_path)

trainer.train()
