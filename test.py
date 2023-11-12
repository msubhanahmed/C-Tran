import matplotlib.pyplot as plt
from models import CTranModel
from pytorch_grad_cam.utils.image import preprocess_image
import torch
import numpy as np
import cv2 as cv
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



print("Initalizing...")
num_labels = 5
use_lmt = False
device = torch.device('cpu')
pos_emb = False

layers = 3
heads = 4
dropout = 0.1
no_x_features = False



model_path = 'best_model-vgg600.pt'


def reshape_transform(tensor, height=12, width=12):
    tensor = tensor.transpose(0, 1)
    result = tensor[:, :-num_labels, :].reshape(tensor.size(0),height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_saved_model(saved_model_name, model):
    checkpoint = torch.load(saved_model_name,map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if 'densenet' in saved_model_name:
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model

print("Loading Model...")
#model = CTranModel(5, use_lmt, device,'densenet', pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)
model = CTranModel(5, use_lmt, device,'vgg16', pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)
model = load_saved_model(model_path, model)
model.eval()
#print(model.self_attn_layers)

print("Preparing...")
classes={'N':0,'H':1,'M':2,'G':3,'D':4}
predictions = []
labels = []
for i in os.listdir("validation"):
    print(i)
    for j in os.listdir("validation/"+i):
        #rgb_img = cv.imread(image_path, 1)[:, :, ::-1]
        img = cv.imread("validation/"+i+"/"+j)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        laplacian = cv.Laplacian(gray_image, cv.CV_64F)
        threshold_value = 1
        _, final_connected_edges = cv.threshold(np.uint8(np.abs(laplacian)), threshold_value, 255, cv.THRESH_BINARY)
        coords = np.column_stack(np.where(final_connected_edges == 255))
        x, y, w, h = cv.boundingRect(coords)
        cropped_image = img[x:x+w, y:y+h]

        rgb_img = cv.resize(cropped_image, (600, 600))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_in = torch.zeros(1, 5)
        pred = torch.sigmoid_(model(input_tensor.to(device), mask_in.to(device))).detach().cpu()
        predictions.append(torch.argmax(pred[0]))
        labels.append(classes[i])
print(predictions)
print(labels)

conf_matrix = confusion_matrix(labels, predictions)
print("Classification Report:")
print(classification_report(labels, predictions))
print("\nConfusion Matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()