import numpy as np
import timm
import cv2
from pytorch_grad_cam.utils.image import preprocess_image
import torch.nn as nn
import torch

print(timm.list_models('resnet*', pretrained=True))


from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


model = timm.create_model('resnet50', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.eval()

image_path = 'kaz5.PNG'

rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

print(model)

preds = model(input_tensor)
resnet_features = FeatureExtractor(model, layers=["layer4", "global_pool"])
features = resnet_features(input_tensor)
torch.sigmoid_(preds)

class_idxs = np.argsort(-preds[0])

class_idx = class_idxs[0] #np.argmax(preds[0])
print(preds)
#class_idx = 829
print(class_idx)
print(preds[0][class_idx])

print({name: output.shape for name, output in features.items()})
class_weights = model.fc.weight[class_idx]

print(class_weights.shape)
print(features['layer4'].squeeze(0).shape)

bz, nc, h, w = features['layer4'].shape

cam = torch.matmul(class_weights, features['layer4'].reshape(nc, h*w))
print(cam.shape)
cam = cam.reshape(h, w)
cam = cam - cam.min()
cam_img = cam / cam.max()
cam_img = np.uint8(255 * cam_img)
cam_img = cv2.resize(cam_img, (224, 224))

heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

original_img = cv2.imread(image_path)
original_img = cv2.resize(original_img, (224, 224))

result = heatmap.astype('float16') * 0.5 + original_img.astype('float16') * 0.5

print(heatmap.dtype)
print(original_img.dtype)
print(result.dtype)

cv2.imshow('original', original_img)
cv2.imshow('heatmap', heatmap)
cv2.imshow(str(class_idx), result.astype('uint8'))
cv2.waitKey(0)