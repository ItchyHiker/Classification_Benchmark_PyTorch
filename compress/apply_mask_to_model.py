import os, sys
sys.path.append('.')
import time

import nni
import torch

from nets.mobilenet_v2_relu import MobileNetV2
import config

from nni.compression.torch import apply_compression_results
from nni.compression.speedup.torch import ModelSpeedup

dummy_input = torch.randn((64, 3, 224, 224)).cuda()
model = MobileNetV2(n_class=config.num_classes, width_mult=1.0)
model.cuda()

start = time.time()
for i in range(32):
    output = model(dummy_input)
end = time.time()
print("Time for original model:", end - start)

model.load_state_dict(torch.load('results/pruned/pruned_model.pth'))
mask_file = './results/pruned/pruned_mask.pth'

apply_compression_results(model, mask_file, 'cuda')

start = time.time()
for i in range(32):
    mask_output = model(dummy_input)
end = time.time()
print("Time for masked model:", end - start)

m_speedup = ModelSpeedup(model, dummy_input, mask_file, 'cuda')
m_speedup.speedup_model()

start = time.time()
for i in range(32):
    speedup_output = model(dummy_input)
end = time.time()
print("Time for speedup model:", end - start)

