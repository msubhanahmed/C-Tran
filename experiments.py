import timm
import numpy as np
import torch
import math

from models import swin, swin_original, CTranModel, CTran_original

import torchvision.models as models

#model = models.resnext101_32x8d(pretrained=False)
#print(model)

#print(timm.list_models('*swin*', pretrained=True))
#model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)

#print(model)

test_img = torch.from_numpy(np.zeros((1, 3, 224, 224))).float()
print(test_img.shape)
test_mask = torch.from_numpy(np.zeros((1, 20))).float()


#ctran_or = CTran_original.CTranModel(backbone_model='test', device='cpu', num_labels=20, use_lmt=True)
#ctran = CTranModel(backbone_model='resnet101', device='cpu', num_labels=20, use_lmt=True)

#emb, _, _ = ctran_or.forward(test_img, test_mask)
#features, label_emb, _ = ctran.forward(test_img, test_mask)

#print('embeddings', emb.shape)

# make embeddings perfect square
#padding = int(math.sqrt(emb.shape[1])) + 1
#print('padding', padding)
#padding_size = padding ** 2 - emb.shape[1]
#print('padding_size ', padding_size)

#padding = torch.from_numpy(np.zeros((1, padding_size, emb.shape[2])))
#print('padding shape', padding.shape)

# put the padding first in the tokens
#emb = torch.cat((padding, emb), dim=1)
#print(emb.shape)

#print('ctran emb', emb.shape)

#model = swin.SwinTransformer()
#model.forward(test_img, label_emb)

model_or = swin_original.SwinTransformer()
model_or.forward(test_img)
