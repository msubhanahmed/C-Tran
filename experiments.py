from PIL import Image

import os
import matplotlib.pyplot as plt
import numpy as np

dataset_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training'


smallest_w = 1000000
smallest_h = 1000000
smallest_img = 10000000
smallest_img_w = 1000000
smallest_img_h = 10000000
smallest_img_name = ''


bucket_h = np.zeros(400)
bucket_w = np.zeros(400)

for file in os.listdir(dataset_path):
    img = Image.open(os.path.join(dataset_path, file))
    w, h = img.size

    if w * h < smallest_img:
        smallest_img = w * h
        smallest_img_w = w
        smallest_img_h = h
        smallest_img_name = file

    if w < smallest_w:
        smallest_w = w

    if h < smallest_h:
        smallest_h = h

    bucket_h[h // 10] += 1
    bucket_w[w // 10] += 1


plt.plot(np.arange(135, 150), bucket_h[135:150])
plt.plot(np.arange(135, 150), bucket_w[135:150])

plt.xticks(np.arange(135, 150))
plt.show()

print('smallest w', smallest_w)
print('smallest h', smallest_h)
print('name ', smallest_img_name)
print('smallest img', smallest_img_w, smallest_img_h)