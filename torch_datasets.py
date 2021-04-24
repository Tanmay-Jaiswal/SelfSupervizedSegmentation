import os
import numpy as np
import torch
import torch.utils.data


class TumorSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "patches"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "patches", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = np.load(img_path)
        mask = np.load(mask_path)

        mask = torch.as_tensor(mask, dtype=torch.long)
        img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
            print("Applying Traansforms!!", img.shape, mask.shape)

        return img, mask

    def __len__(self):
        return len(self.imgs)


class TumorRecDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "patches"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "patches"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "patches", self.imgs[idx])
        mask_path = os.path.join(self.root, "patches", self.masks[idx])
        img = np.load(img_path)
        mask = np.load(mask_path)

        mask = torch.as_tensor(mask, dtype=torch.float).permute(2, 0, 1)
        img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
            print(img.shape, mask.shape)

        return img, mask

    def __len__(self):
        return len(self.imgs)
