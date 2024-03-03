import json
import numpy as np
from sklearn.metrics import classification_report


file_path = "data.json"
with open(file_path, "r") as f:
  data = json.load(f)


Treference = ['D', 'G', 'H', 'M', 'N']
Preference = ['N', 'H', 'M', 'G', 'D']

Cpredicted  = []
Vpredicted  = []
CVpredicted = []
actual      = []

for item in data:
    mean_arr = np.mean([item['C-prob'], item['V-prob']], axis=0)
    
    C_label  = np.argmax(item['C-prob'])
    V_label  = np.argmax(item['V-Probs'])
    CV_label = np.argmax(mean_arr)
    Cpredicted.append(Treference[C_label])
    Vpredicted.append(Treference[V_label])
    CVpredicted.append(Treference[CV_label])

    actual.append(Preference[item['label']])

print("CTran")
print(classification_report(actual, Cpredicted))
print("ViT")
print(classification_report(actual, Vpredicted))
print("Ensemble")
print(classification_report(actual, CVpredicted))