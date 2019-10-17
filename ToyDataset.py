from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import itertools
from PIL import Image

class ToyDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        class_paths = [x[0] for x in os.walk(root_dir)]
        self.classes = [x.split("/")[-1] for x in class_paths[1:]]
        self.image_paths = []
        for c in class_paths:
            [self.image_paths.append((c.split("/")[-1],x)) for x in os.listdir(c)]
        self.root_dir = class_paths[0]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx][1]
        label = self.image_paths[idx][0]
        image = Image.open("{}/{}/{}".format(self.root_dir,label,img_name)).convert('RGB')
        label = self.classes.index(label)
        label = torch.from_numpy(np.array([label]).astype('float'))
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample
