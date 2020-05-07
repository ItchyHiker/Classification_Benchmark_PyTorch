import os, sys
sys.path.append('.')

import cv2
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
import numpy as np

import config

class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, scheduler, 
            epochs, device, save_path):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.loss = CrossEntropyLoss().to(self.device)

    def train(self):
        self.model.train()
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                _, pred_labels = torch.max(outputs, 1)
             
                loss = self.loss(outputs, labels)
                acc = torch.sum(pred_labels == labels) / float(len(labels))
                
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

    def test(self, epoch):
        confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        self.model.eval() 
        with torch.no_grad():
            total_acc = 0
            total_sample = 0
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, pred_labels = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                acc = torch.sum(pred_labels == labels)
                total_acc += acc
                total_sample += len(inputs) 
            acc = float(total_acc) / total_sample
        print("Confusion Matrix:", confusion_matrix.diag()/confusion_matrix.sum(1))
        print("Test Acc:", acc)

