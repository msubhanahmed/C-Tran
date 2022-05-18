import timm
import torch
import torchvision.models
import torch.nn as nn
import einops
import math

import torchvision.models as models

#model = models.resnext101_32x8d(pretrained=False)
#print(model)
from models.backbone import InceptionBackbone

model = torchvision.models.wide_resnet101_2(pretrained=False)

test = torch.zeros((1,3,384,384))

print(test.shape)

print(model)

out = model(test)

print(out[0].shape)