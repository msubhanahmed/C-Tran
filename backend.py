from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from flask_cors import CORS, cross_origin
from models import CTranModel
from pytorch_grad_cam.utils.image import preprocess_image
import torch
import cv2 as cv
from skimage.metrics import structural_similarity as ssim

def image_similarity(image1, image2):
    img1_gray = image1.convert('L')
    img2_gray = image2.convert('L')
    img1_array = np.array(img1_gray)
    img2_array = np.array(img2_gray)
    similarity = ssim(img1_array, img2_array)    
    return similarity



print("Initalizing...")
num_labels      = 5
use_lmt         = False
device          = torch.device('cpu')
pos_emb         = False
layers          = 3
heads           = 4
dropout         = 0
no_x_features   = False


model_path      = '/kaggle/input/saved-models/CTran-VGG-P.pt'

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
model = CTranModel(5, use_lmt, device,'vgg16', pos_emb, layers, heads, dropout, no_x_features, grad_cam=True)
model = load_saved_model(model_path, model)
model.eval()
model.to(device)


app = Flask(__name__)
CORS(app) 

@app.route('/upload_photo', methods=['POST'])
@cross_origin()
def upload_photo():
    classes = ['Normal', 'Hypertensive', 'Myopia', 'Galucoma', 'Diabetic']
    try:
        if 'photo' not in request.files:
            return jsonify({'error': 'No photo provided'}), 400

        photo        = request.files['photo']
        pil_image    = Image.open(photo).resize((224, 224))
        image_array  = np.array(pil_image)
        rgb_img      = np.float32(image_array)/255.0
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_in      = torch.zeros(1, 5)

        with torch.no_grad():
            pred     = model(input_tensor.to(device), mask_in.to(device))
        prob         = F.softmax(pred,dim=1).detach().cpu()
        prediction   = torch.argmax(prob.tolist()[0])
        prediction   = int(prediction)
        return jsonify({'prediction': classes[prediction]}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error':"Some Error Occured"}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
    #app.run(debug=True, host='0.0.0.0', port=5000)
