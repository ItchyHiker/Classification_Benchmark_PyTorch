import os, sys
sys.path.append('.')

import cv2
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from nni.compression.torch import SlimPruner, L1FilterPruner, ActivationMeanRankFilterPruner
from nni.compression.torch import apply_compression_results
from nni.compression.speedup.torch import ModelSpeedup

import config
from nets.mobilenet_v2 import MobileNetV2
class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, scheduler, 
            epochs, prune_epochs, device, save_path):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.prune_epochs = prune_epochs
        self.device = device
        self.save_path = save_path

        self.loss = CrossEntropyLoss()
        # prune
        self.config_list = [{'sparsity': 0.1, 'op_types': ['Conv2d']}]

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, pred_labels = torch.max(outputs, 1)
             
                loss = self.loss(outputs, labels)
                acc = torch.sum(pred_labels == labels.data) / float(len(labels))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if batch_idx % 100 == 0:
                    print("Train Epoch: {:03} [{:05}/{:05} ({:03.0f}%) \t Loss:{:.6f} Acc:{:.6f} LR: {:.6f}".format(epoch, 
                                            batch_idx*len(inputs), 
                                            len(self.train_loader.dataset),
                                            100.*batch_idx/len(self.train_loader),
                                            loss.item(),
                                            acc,
                                            self.optimizer.param_groups[0]['lr']))
            self.scheduler.step()
            torch.save(self.model.state_dict(), 
                    os.path.join(self.save_path, '{}_mobilenetv2_epoch_{}.pth'.format(
                        config.attribute, epoch)))
            self.test(epoch) 
        
        self.prune()
                # apply_compression_results(self.model, 'results/pruned/pruned_mask.pth', None)
                # speedup_model = ModelSpeedup(model. torch.randn(1, 3, 224, 224).cuda(),
                #        'results/pruned/pruned_mask.pth', None)
                # speedup_model.speedup_model()
                # torch.save(model.state_dict(), 'pruned_speedup_model.pth')
    def prune(self):
        self.pruner = ActivationMeanRankFilterPruner(self.model, self.config_list,
                self.optimizer)
        self.model = self.pruner.compress()
        top_acc = 0.9
        for epoch in range(self.prune_epochs):
            self.pruner.update_epoch(epoch)
            self._train_one_epoch(epoch, self.model, self.train_loader, self.optimizer)
            acc = self.test(epoch)
            if acc > top_acc:
                top_acc = acc
                print("Begining prune model")
                self.pruner.export_model(model_path='results/pruned/pruned_model.pth',
                        mask_path='results/pruned/pruned_mask.pth')
                

    def _train_one_epoch(self, epoch, model, train_loader, optimizer):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, pred_labels = torch.max(outputs, 1)

            loss = self.loss(outputs, labels)
            acc = torch.sum(pred_labels == labels.data) / float(len(labels))

            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("Train Epoch: {:03} [{:05}/{:05} ({:03.0f}%) \t Loss:{:.6f} Acc:{:.6f} LR: {:.6f}".format(epoch, 
                                            batch_idx*len(inputs), 
                                            len(self.train_loader.dataset),
                                            100.*batch_idx/len(self.train_loader),
                                            loss.item(),
                                            acc,
                                            self.optimizer.param_groups[0]['lr']))
    
    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0
            total_sample = 0
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, pred_labels = torch.max(outputs, 1)
                
                acc = torch.sum(pred_labels == labels.data)
                total_acc += acc
                total_sample += len(inputs) 
            acc = float(total_acc) / total_sample
    
        print("Test Acc:", acc)
        return acc
