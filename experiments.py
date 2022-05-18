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

model = torchvision.models.inception_v3(pretrained=False, init_weights=True)

test = torch.zeros((1,3,384,384))

print(test.shape)

print(model)

model.avgpool = nn.Identity()
model.dropout = nn.Identity()
model.fc = nn.Identity()

out = model(test)

print(out.logits.shape)