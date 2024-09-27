import os
import shutil
import random

def split_dataset(source_folder, masks_folder, train_folder, val_folder, test_folder, masks_train_folder, masks_val_folder, masks_test_folder, train_size=500, test_size=55, val_size=145):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(masks_train_folder, exist_ok=True)
    os.makedirs(masks_val_folder, exist_ok=True)
    os.makedirs(masks_test_folder, exist_ok=True)

    image_files = sorted(os.listdir(source_folder))[:700]  # Get only the first 700 images
    mask_files = sorted(os.listdir(masks_folder))[:700]  # Get corresponding mask files

    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files, mask_files = zip(*combined)

    train_images = image_files[:train_size]
    train_masks = mask_files[:train_size]
    
    test_images = image_files[train_size:train_size + test_size]
    test_masks = mask_files[train_size:train_size + test_size]
    
    val_images = image_files[train_size + test_size:]
    val_masks = mask_files[train_size + test_size:]

    def move_files(images, masks, dest_img_folder, dest_mask_folder):
        for img, mask in zip(images, masks):
            shutil.copy(os.path.join(source_folder, img), os.path.join(dest_img_folder, img))
            shutil.copy(os.path.join(masks_folder, mask), os.path.join(dest_mask_folder, mask))

    move_files(train_images, train_masks, train_folder, masks_train_folder)
    move_files(test_images, test_masks, test_folder, masks_test_folder)
    move_files(val_images, val_masks, val_folder, masks_val_folder)

    print(f"Split completed! \nTrain: {train_size}, Test: {test_size}, Val: {val_size}")

source_folder = "data/train/train"
masks_folder = "data/train_masks/train_masks/"
train_folder = "data/split/train_images/"
val_folder = "data/split/val_images/"
test_folder = "data/split/test_images/"
masks_train_folder = "data/split/train_masks/"
masks_val_folder = "data/split/val_masks/"
masks_test_folder = "data/split/test_masks/"

split_dataset(source_folder, masks_folder, train_folder, val_folder, test_folder, masks_train_folder, masks_val_folder, masks_test_folder)
