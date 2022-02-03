import timm

import torchvision.models as models

model = models.resnext101_32x8d(pretrained=False)
print(model)

print(timm.list_models('*resnext101*', pretrained=True))
#model = timm.create_model('resnet101', pretrained=False)

#print(model)
