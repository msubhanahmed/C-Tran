import timm
import numpy as np
import torch
import math

from models import swin, swin_original, CTranModel, CTran_original

import torchvision.models as models

print(timm.list_models('*swin*', pretrained=True))
#model = timm.create_model('resnet50', pretrained=False)

#print(model)