import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from PIL import Image
import numpy as np
import torch
import h5py


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, image_size, mode='train'):
        super(HDF5Dataset, self).__init__()
        with h5py.File(file_path, "r") as hf:
            self.data = np.asarray(hf['images'])
            self.target = (np.expand_dims(np.asarray(hf['labels']), -1) - 1)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1/256., 1/256., 1/256.])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[1/256., 1/256., 1/256.])
            ])

    def __getitem__(self, index):
        x = np.array(self.data[index, :, :, :], dtype='uint8')

        x = self.transform(x)

        y = torch.from_numpy(
            np.asarray(self.target[index], dtype=np.int16)).float()

        return x, y

    def __len__(self):
        return list(self.data.shape)[0]
