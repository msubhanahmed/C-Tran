import numpy as np
import timm
import cv2
from pytorch_grad_cam.utils.image import preprocess_image
import torch.nn as nn
import torch

from typing import Dict, Iterable, Callable

from models import CTranModel


def reshape_transform(tensor, height=12, width=12):
    #return tensor
    print('to_reshape', tensor.shape)
    tensor = tensor.transpose(0, 1)
    print('transpose', tensor.shape)
    result = tensor[:, :-20, :].reshape(tensor.size(0),
                                               height, width, tensor.size(2))

    print('middle step', result.shape)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    print('reshape_result', result.shape)
    return result


def load_saved_model(saved_model_name, model):
    checkpoint = torch.load(saved_model_name)
    state_dict = checkpoint['state_dict']

    if 'densenet' in saved_model_name:
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = True

    return model


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
num_labels = 5
use_lmt = False
backbone_model = 'densenet'
device = torch.device('cuda')
pos_emb = False

layers = 3
heads = 4
dropout = 0.1
no_x_features = False
model_path = '/kaggle/input/saved-models/best_model.pt'

model = CTranModel(num_labels, use_lmt, device, backbone_model, pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)
model = load_saved_model(model_path, model)
#print(model)

model.eval()
model.cuda()
print(dict([*model.named_modules()]))

# Read and prepare image
image_path = input("Image Path: ")
#image_path = 'C:\\Users\\AI\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop\\im0045.png'
#image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop\\aria_d_15_22.tif'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (384, 384))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# Get features and predictions
preds = torch.sigmoid_(model(input_tensor.cuda()))
ctran_features = FeatureExtractor(model, layers=["self_attn_layers.2.transformer_layer.norm1"])#'backbone.base_network.layer4.2.bn3'])#'backbone.base_network.avgpool']) #'"self_attn_layers.2.transformer_layer.norm1"])
features = ctran_features(input_tensor.cuda())

print(preds)
#print(features)
activation_maps = reshape_transform(torch.mean(torch.stack(tuple(features.values())), dim=0))
print(activation_maps.shape)

# Get best class prediction
class_idxs = np.argsort(-preds[0].detach().cpu())

class_idx = class_idxs[0] #np.argmax(preds[0])
#print(preds)
#class_idx = 829
print(class_idx)

#print(model)

bz, nc, h, w = activation_maps.shape
class_weights = model.output_linear.weight[class_idx]

print(class_weights.shape)
print(activation_maps.shape)

cam = torch.matmul(class_weights, activation_maps.reshape(nc, h*w)).detach().cpu()
print(cam.shape)
cam = cam.reshape(h, w)
cam = cam - cam.min()
cam_img = cam / cam.max()
cam_img = np.uint8(255 * cam_img)
cam_img = cv2.resize(cam_img, (384, 384))

heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

original_img = cv2.imread(image_path)
original_img = cv2.resize(original_img, (384, 384))

result = heatmap.astype('float16') * 0.5 + original_img.astype('float16') * 0.5

print(heatmap.dtype)
print(original_img.dtype)
print(result.dtype)

cv2.imshow('original', original_img)
cv2.imshow('heatmap', heatmap)
cv2.imshow(str(class_idx), result.astype('uint8'))
cv2.waitKey(0)