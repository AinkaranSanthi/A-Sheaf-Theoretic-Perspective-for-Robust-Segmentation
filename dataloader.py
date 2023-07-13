import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os
# from numpy.core.numeric import full
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from pandas.core.dtypes import base
import torch
from tqdm import tqdm
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
# import pickle
from torch.utils.data.dataset import Dataset
import pytorch_lightning
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import math
import torchvision
from skimage import measure
from monai.losses import DiceCELoss

from monai.metrics import DiceMetric,compute_meandice,  compute_hausdorff_distance, compute_average_surface_distance
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)
from monai.transforms  import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    NormalizeIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
    RandRotated,
    RandAffined,
    RandGaussianSharpend,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandRotate90d,
    RandBiasFieldd,
    ToTensord,
    ScaleIntensityd
)
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
#import torchio as tio
from PIL import Image
import subprocess
sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

print('Last run on', time.ctime())

class ProstateMRIDataset(Dataset):
    def __init__(self, csv_file_img, binarise, datatype, contrastive,image, label, image_size):
        #self.data = csv_file_img
        self.data = pd.read_csv(csv_file_img)
        self.datatype = datatype
        self.binarise = binarise
        self.image = image
        self.image_size = image_size
        self.label = label
        self.contrastive = contrastive
        self.image_size = image_size
        self.train_transforms = Compose(
             [
                LoadImaged(
                     allow_missing_keys = True,keys = ["t2image", "adcimage", "t2labels", "adclabels"]
                ),
                AddChanneld( allow_missing_keys = True,keys=["t2image", "adcimage", "t2labels", "adclabels"]),
                Orientationd(allow_missing_keys = True, keys=["t2image", "adcimage", "t2labels", "adclabels"], axcodes="RAS"),
                NormalizeIntensityd(allow_missing_keys = True,  keys=["t2image", "adcimage"], nonzero=True, channel_wise=True),
                SpatialPadd(
                    allow_missing_keys = True,
                    keys=["t2image", "adcimage", "t2labels", "adclabels"],
                    spatial_size= (self.image_size),
                ),
                CenterSpatialCropd(
                    allow_missing_keys = True,
                    keys=["t2image", "adcimage", "t2labels", "adclabels"],
                    roi_size=(self.image_size),
                ),
                RandAffined(
                    allow_missing_keys = True,
                    keys=["t2image", "adcimage", "t2labels", "adclabels"],
                    prob=0.2,
                ),
                RandShiftIntensityd(allow_missing_keys = True,keys=["t2image", "adcimage"], offsets=0.1, prob=0.0),
                RandGaussianNoised(allow_missing_keys = True,keys = ["t2image", "adcimage"], std = 0.1, prob = 0.0),
                RandGaussianSharpend(allow_missing_keys = True,keys = ["t2image", "adcimage"],  prob = 0.0),
                RandAdjustContrastd(allow_missing_keys = True,keys=["t2image", "adcimage"],prob = 0.0),
                RandBiasFieldd(allow_missing_keys = True, keys=["t2image", "adcimage"],   prob=0.0),

             ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(
                    allow_missing_keys = True,
                    keys = ["t2image", "t2labels"],
                ),
                AddChanneld(allow_missing_keys = True,keys=["t2image", "t2labels"]),
                Orientationd(allow_missing_keys = True,keys=["t2image",  "t2labels"],as_closest_canonical=True, axcodes="RAS"),
                NormalizeIntensityd(allow_missing_keys = True,  keys=["t2image"], nonzero=True, channel_wise=True),
                SpatialPadd(
                    allow_missing_keys = True,
                    keys=["t2image",  "t2labels"],
                    spatial_size= (self.image_size),
                ),
                CenterSpatialCropd(
                    allow_missing_keys = True,
                    keys=["t2image", "t2labels"],
                    roi_size=(self.image_size),
                ),
            ]
        )
        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            sample = {}
            if self.image:
               img_patht2 = self.data.loc[idx, 't2image']
               sample['t2image'] = img_patht2
               img_pathadc = self.data.loc[idx, 'adcimage']
               sample['adcimage'] = img_pathadc
            if self.label:
               img_label = self.data.loc[idx, 't2label']
               sample['t2labels'] = img_label

            self.samples.append(sample)

    def get_preprocessing_transform(self):
        #self.img_size = self.get_max_shape(self.subjects + self.test_subjects)
        self.img_size = [256, 256, 24]
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad([256, 256, 24]),
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.get_sample(item)
        image = {}
        if (self.image == True and self.label == False):
            if self.contrastive  == True:
              t2image =  sample['t2image']
              adcimage =  sample['adcimage']
            else:
              t2image =  sample['t2image']
        elif (self.label == True and self.image == False):
            if self.contrastive  == True: 
              t2label  = sample['t2labels']
              adclabel  = sample['adclabels']
              t2label  = torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label
              adclabel  = torch.where(adclabel > 0, 1, 0) if self.binarise == True else adclabel
      
            else:
              t2label  = sample['t2labels']
              t2label  = torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label

        else:
            if self.contrastive  == True: 
               t2label = sample['t2labels']
               t2image =  sample['t2image']
               adcimage =  sample['adcimage']
               t2label  = torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label
       
            else:
               t2label = sample['t2labels']
               t2image =  sample['t2image']
               t2label  = torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label
    

        if self.contrastive  == True:        
            image = {'t2image': t2image,'adcimage':adcimage,  't2labels':  torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label}
        else:
            image = {'t2image': t2image,  't2labels':  torch.where(t2label > 0, 1, 0) if self.binarise == True else t2label}
        return image

    def get_sample(self, item):
        sample = self.samples[item]
        
        if self.datatype == 'train':
            sample = self.train_transforms(sample)
        else:
            sample = self.val_transforms(sample)
        
        return sample
class ProstateMRIDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, binarise, image, label,batch_size, num_workers, contrastive, image_size):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size =batch_size
        self.num_workers = num_workers


        self.train_set = ProstateMRIDataset(csv_file_img = self.csv_train_img, binarise=binarise, image=image, label=label,contrastive = contrastive, image_size = image_size, datatype='train')
        self.val_set = ProstateMRIDataset(csv_file_img = self.csv_val_img, binarise=binarise, image=image, label=label, contrastive = False, image_size = image_size, datatype= 'val')
        self.test_set = ProstateMRIDataset(csv_file_img = self.csv_test_img, binarise=binarise, image=image, label=label, contrastive = False, image_size = image_size,datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
