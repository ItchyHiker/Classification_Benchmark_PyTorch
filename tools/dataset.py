import os, sys
sys.path.append('.')
import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, models, transforms

import config

class Dataset(object):
    def __init__(self, image_path, is_train=False):
        self.image_path = image_path
        self.is_train = is_train
        if self.is_train:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.Resize([config.image_size, config.image_size]),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomAffine(degrees=10, 
                                        translate=(0.2, 0.2), 
                                        scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize([config.image_size, config.image_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def load_data(self):
        dataset = datasets.ImageFolder(self.image_path, self.transforms) 
        '''
        if self.is_train:
            weights = self._calculate_weights_for_balanced_classes(dataset.imgs, 
                len(dataset.classes))
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                shuffle=False, sampler=sampler, num_workers=8)
        else:
        '''
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                shuffle=True, num_workers=8)
        return data_loader
    
    def _calculate_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    dataset = Dataset('./data/flowers_102/train', True)
    data_loader = dataset.load_data()
    for data, label in data_loader:
        data = data[0]
        img = data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
