import json
import numpy as np
from sklearn.metrics import classification_report
import torch

file_path = "data.json"
with open(file_path, "r") as f:
  data = json.load(f)
print("Total Items: ",len(data))

Treference = ['D', 'G', 'H', 'M', 'N']
Preference = ['N', 'H', 'M', 'G', 'D']

Cpredicted  = []
Vpredicted  = []
CVpredicted = []
actual      = []

for item in data:
    c_logits_tensor = torch.tensor(item['C-logits'])
    sigmoid_C_logits = torch.sigmoid(c_logits_tensor).numpy()
    mean_arr = np.average([sigmoid_C_logits[::-1], item['V-Probs']], axis=0, weights=[0.5, 0.6])
    #mean_arr = np.average([item['C-prob'][::-1], item['V-Probs']], axis=0,weights=[0.5, 0.5])
    C_label  = np.argmax(sigmoid_C_logits > 0.5)#np.argmax(item['C-prob'])
    V_label  = np.argmax(item['V-Probs'])
    CV_label = np.argmax(mean_arr)
    Cpredicted.append(Preference[C_label])
    Vpredicted.append(Treference[V_label])
    CVpredicted.append(Treference[CV_label])

    actual.append(Preference[item['label']])

print("CTran")
print(classification_report(actual, Cpredicted,target_names=Treference))
print("ViT")
print(classification_report(actual, Vpredicted,target_names=Treference))
print("Ensemble")
print(classification_report(actual, CVpredicted,target_names=Treference))