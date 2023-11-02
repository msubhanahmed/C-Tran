import torch
import os
import pandas as pd
import numpy as np

from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices
import cv2 as cv


class MsaDataset(torch.utils.data.Dataset):
    def __init__(self, data, img_dir='', image_transform=None, known_labels=0, testing=False):

        self.img_dir = img_dir
        self.img_names = data['Name'].values.astype(str)
        self.labels = data.iloc[:, 1:].to_numpy(dtype=np.float)
        self.num_labels = 5
        self.known_labels = known_labels
        self.testing = testing
        self.image_transform = image_transform
        self.epoch = 1

    def __getitem__(self, index):
        name = self.img_names[index]
        img = cv.imread(name)
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        laplacian = cv.Laplacian(gray_image, cv.CV_64F)
        threshold_value = 1
        _, final_connected_edges = cv.threshold(np.uint8(np.abs(laplacian)), threshold_value, 255, cv.THRESH_BINARY)
        coords = np.column_stack(np.where(final_connected_edges == 255))
        x, y, w, h = cv.boundingRect(coords)
        cropped_image = img[x:x+w, y:y+h]
        image = Image.fromarray(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))

        if self.image_transform:
            image = self.image_transform(image)
            image = image['image']

        labels = torch.Tensor(self.labels[index])

        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels, self.epoch)

        mask = labels.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': name,
        }

        # print(sample)

        return sample

    def __len__(self):
        return len(self.img_names)
