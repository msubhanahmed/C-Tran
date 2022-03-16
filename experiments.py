import timm

import torchvision.models as models

#model = models.resnext101_32x8d(pretrained=False)
#print(model)

print(timm.list_models('*convnext*', pretrained=True))
model = timm.create_model('convnext_base_in22k', pretrained=False)

print(model)