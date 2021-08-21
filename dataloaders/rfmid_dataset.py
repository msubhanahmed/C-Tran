import torch
import os
import pandas as pd
import numpy as np

from PIL import Image
from torch._C import dtype
from dataloaders.data_utils import get_unk_mask_indices


class RFMiDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir='./data/...', anno_path='checar path csv', image_transform=None, labels_path='creo que es el mismo que el csv', known_labels=0, testing=False):
        data = pd.read_csv(anno_path)
        
        self.img_dir = img_dir
        self.img_names = data.loc[:, 'ID'].values.astype(str)

        self.num_labels = 28
        self.known_labels = known_labels
        self.testing = testing

        self.labels = np.array(data.iloc[:, 2:])
        self.image_transform = image_transform
        self.epoch = 1

    def __getitem__(self, index):
        name = self.img_names[index] + '.png'
        image = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        
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

        #print(sample)

        return sample

    def __len__(self):
        return len(self.img_names)

    def get_labels(self):
        return self.labels
