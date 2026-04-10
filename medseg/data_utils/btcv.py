import os 
import glob
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import argparse
from transforms import build_btcv_train_transforms, build_btcv_test_transforms

def build_btcv_dataloader(args,mode: str = "train"):
    data_path = args.data_path

    #Data paths
    print(data_path)
    train_images = sorted(glob.glob(os.path.join(data_path, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_path, "labelsTr", "*.nii.gz")))
    test_images = sorted(glob.glob(os.path.join(data_path, "imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(data_path, "labelsTs", "*.nii.gz")))
    print(train_images)
    train_files = [{"image": image, "label": label} for image, label in zip(train_images, train_labels)]
    test_files = [{"image": image, "label": label} for image, label in zip(test_images, test_labels)]
    
    if mode == "train":
        train_transforms = build_btcv_train_transforms()
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
        return train_loader
    else:
        test_transforms = build_btcv_test_transforms()
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BTCV Dataloader")
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument("--data_path", type=str, default=os.path.join(_project_root, "data", "BTCV"), help="Path to BTCV data")
    args = parser.parse_args()
    train_loader = build_btcv_dataloader(args, mode="train")
    for batch_data in train_loader:
        image, labels = batch_data["image"], batch_data["label"]
        print(image.shape)
        print(labels.shape)
