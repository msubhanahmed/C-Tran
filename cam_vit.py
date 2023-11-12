import numpy as np
import timm
import cv2
from pytorch_grad_cam.utils.image import preprocess_image
import torch.nn as nn
import torch

from typing import Dict, Iterable, Callable


def reshape_transform(tensor, height=24, width=24):
    print('to_reshape', tensor.shape)
    #tensor = tensor.transpose(0, 1)
    print('transpose', tensor.shape)
    result = tensor[:, 1 :, :].reshape(tensor.size(0),
                                               height, width, tensor.size(2))

    print('middle step', result.shape)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    print('reshape_result', result.shape)
    return result


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


# Create model
model = timm.create_model('vit_base_patch16_384', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.eval()
print(dict([*model.named_modules()]))

# Read and prepare image
image_path = '/kaggle/input/fyp-dataset/validation/D/107_right.jpg'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (384, 384))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# Get features and predictions
preds = torch.sigmoid_(model(input_tensor))
vit_features = FeatureExtractor(model, layers=["blocks.11.norm1"])
features = vit_features(input_tensor)

print(features)
activation_maps = reshape_transform(torch.mean(torch.stack(tuple(features.values())), dim=0))
print(activation_maps.shape)

# Get best class prediction
class_idxs = np.argsort(-preds[0])

class_idx = class_idxs[0] #np.argmax(preds[0])
#print(preds)
#class_idx = 829
print(class_idx)

bz, nc, h, w = activation_maps.shape
class_weights = model.head.weight[class_idx]

cam = torch.matmul(class_weights, activation_maps.reshape(nc, h*w))
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