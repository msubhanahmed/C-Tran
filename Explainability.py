from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import CTranModel
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.ablation_layer import AblationLayer
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torch

import numpy as np
import cv2

num_labels = 20
use_lmt = False
device = torch.device('cuda')
pos_emb = False

layers = 3
heads = 4
dropout = 0.1
no_x_features = False

image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training\\519.png'
#image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop\\aria_a_13_2.tif'
#image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop\\im0264.png'
model_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\codes\\trained_models\\c-tran\\best_model.pt'


def reshape_transform(tensor, height=14, width=14):
    #print('to_reshape', tensor.shape)
    tensor = tensor.transpose(0, 1)
    #print('transpose', tensor.shape)
    result = tensor[:, :-num_labels, :].reshape(tensor.size(0),
                                               height, width, tensor.size(2))

    #print('middle step', result.shape)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    #print('reshape_result', result.shape)
    return result


def load_saved_model(saved_model_name, model):
    checkpoint = torch.load(saved_model_name)
    model.load_state_dict(checkpoint['state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    return model


model = CTranModel(num_labels, use_lmt, device,'tv_resnet101', pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)

model = load_saved_model(model_path, model)
#print(model.self_attn_layers)

rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (448, 448))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

mask_in = torch.zeros(1, 20)

model.eval()
model.cuda()

pred = torch.sigmoid_(model(input_tensor.to(device), mask_in.to(device))).detach().cpu()
print(pred)

target_layers = [model.self_attn_layers[-1].transformer_layer.norm1]

cam = ScoreCAM(model=model,
               target_layers=target_layers,
               use_cuda=True,
               reshape_transform=reshape_transform)
# ablation_layer=AblationLayerVit)

target_class = 4
print(pred[0, target_class].item())

targets = [ClassifierOutputTarget(target_class)]

cam.batch_size = 1

grayscale_cam = cam(input_tensor=input_tensor,
                   targets=targets,
                   eigen_smooth=False,
                   aug_smooth=False)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam)

img_name = image_path.split('\\')[-1].split('.')[0]

cv2.imwrite(img_name + '_' + str(target_class) + '_' + "{:.3f}".format(pred[0, target_class].item()) + '.jpg', cam_image)