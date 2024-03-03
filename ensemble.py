import numpy as np
from PIL import Image
from models import CTranModel
from pytorch_grad_cam.utils.image import preprocess_image
import torch
import cv2 as cv
import pandas as pd
import json
import os

print("Initalizing...")
# ------------------ CTran Intilization ------------------ #
num_labels      = 5
use_lmt         = False
device          = torch.device('cuda:0')
pos_emb         = False
layers          = 3
heads           = 4
dropout         = 0
no_x_features   = False

model_path      = '/kaggle/input/saved-models/CTran-VGG-P.pt'
data_root       = "/kaggle/input/fyp-dataset-list"

def reshape_transform(tensor, height=12, width=12):
    tensor = tensor.transpose(0, 1)
    result = tensor[:, :-num_labels, :].reshape(tensor.size(0),height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_saved_model(saved_model_name, model):
    checkpoint = torch.load(saved_model_name,map_location=torch.device('cuda'))
    state_dict = checkpoint['state_dict']
    if 'densenet' in saved_model_name:
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model

print("Loading Model...")
model = CTranModel(5, use_lmt, device,'vgg16', pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)
model = load_saved_model(model_path, model)
model.eval()
model.to(device)

# ------------------ ViT Intilization ------------------ #

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
import torch.nn.functional as F

model_directory = "/kaggle/input/saved-models/ViT/ViT"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_directory)
ViTmodel = ViTForImageClassification.from_pretrained(model_directory)
transform = transforms.Compose([ transforms.Resize((224, 224)),transforms.ToTensor(), ])
ViTmodel.eval()
ViTmodel.to(device)
# ------------------ Dataset Intilization ------------------ #
img_path    = os.path.join(data_root,      'images')
labels_path = os.path.join(data_root,   'file_list.csv')
data = pd.read_csv(labels_path)

output = []

for i in data.iterrows():
    print(f"\r Filename: {i[1]['Name']}" , end="")
    pil_image = Image.open(i[1]['Name']).resize((224, 224))
    image_array = np.array(pil_image)
    rgb_img = np.float32(image_array)/255.0

    # ------------------ Ctran Inference ------------------ #

    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mask_in = torch.zeros(1, 5)
    with torch.no_grad():
        pred = model(input_tensor.to(device), mask_in.to(device))
    prob = torch.sigmoid_(pred).detach().cpu()

    # ------------------ ViT Inference ------------------ #
    
    
    new_root  = "/kaggle/input/fyp-ii-preprocessed/Dataset"
    new_name  = i[1]['Name'].split("/")[4:]
    for j in new_name:
        new_root += "/" + j
    print(f"\r Filename: {new_root}" , end="")
    pil_image = Image.open(new_root).resize((224, 224))
    input_image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = ViTmodel(input_image)
    Vprob = F.softmax(outputs.logits,dim=1).detach().cpu() #torch.argmax(outputs.logits, dim=1).item()

    output.append({
        "C-logits": pred.detach().cpu().tolist()[0],
        "C-prob"  : prob.tolist(),
        "V-logits": outputs #.logits.detach().cpu().tolist()[0],
        "V-Probs" : Vprob.tolist(),
        "label"   : int(np.argmax(i[1][1:].values))})
    break

file_path = "data.json"
with open(file_path, "w") as json_file:
    json.dump(output, json_file, indent=4)