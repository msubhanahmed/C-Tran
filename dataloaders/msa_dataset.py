import torch
import os
import pandas as pd
import numpy as np

from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices


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
        image = Image.open(name).convert('RGB')

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
