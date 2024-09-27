import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # Mask is grayscale, use 'L'

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = mask.squeeze(0)  

        return image, mask


#transformations for images and masks
def get_transforms():
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.long()),  
    ])
    
    return image_transform, mask_transform


def get_train_val_loaders(batch_size=8, num_workers=4):
    image_transform, mask_transform = get_transforms()

    train_dataset = CarvanaDataset(
        image_dir="data/split/train_images/",
        mask_dir="data/split/train_masks/",
        transform=image_transform,
        mask_transform=mask_transform
    )

    val_dataset = CarvanaDataset(
        image_dir="data/split/val_images/",
        mask_dir="data/split/val_masks/",
        transform=image_transform,
        mask_transform=mask_transform
    )

    test_dataset = CarvanaDataset(
        image_dir="data/split/test_images/",
        mask_dir="data/split/test_masks/",
        transform=image_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

