import torch
import os

import pandas as pd
import numpy as np

from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, data, img_dir=[], image_transform=None, known_labels=0, testing=False):

        #data = pd.read_csv(anno_path)

        self.img_dir = img_dir
        self.img_names = data.loc[:, 'ID'].values.astype(str)
        self.img_dataset = data.iloc[:, 1:4].values.astype(int)

        self.num_labels = 20
        self.known_labels = known_labels
        self.testing = testing

        self.labels = np.array(data.iloc[:, 4:])
        self.image_transform = image_transform
        self.epoch = 1

    def __getitem__(self, index):
        if self.img_dataset[index, 0] == 1: #ARIA
            name = self.img_names[index] + '.tif'
            dataset_idx = 0
        elif self.img_dataset[index, 1] == 1: #STARE
            name = self.img_names[index] + '.png'
            dataset_idx = 1
        else: #RFMiD
            name = self.img_names[index] + '.png'
            dataset_idx = 2

        image = Image.open(os.path.join(self.img_dir[dataset_idx], name)).convert('RGB')

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

        return sample


    def __len__(self):
        return len(self.img_names)
    
    def get_labels(self):
        return self.labels