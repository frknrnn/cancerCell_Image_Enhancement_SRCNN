# ///////////////////////////////////////////////////////////////
# BY: FURKAN EREN
# 01/11/2021
# V: 1.0.0
# ///////////////////////////////////////////////////////////////

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, files):
        mean = np.array([0.485])
        std = np.array([0.229])
        # Transforms for low resolution images and high resolution images

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((33, 33), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((33, 33), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files = files

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')
        img1 = img.resize((10, 10), Image.ANTIALIAS)
        img_lr = self.lr_transform(img1)
        img_hr = self.hr_transform(img)

        return img_lr,img_hr

    def __len__(self):
        return len(self.files)