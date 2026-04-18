import argparse
from typing import Any, List

def load_args():
    parser = argparse.ArgumentParser(description="Train a 3D medical image segmentation model.")
    #dataset 
    parser.add_argument("--dataset", type=str, choices=["btcv", "msd"], required=True, help="Dataset to use for training.")
    parser.add_argument("--data_path", type=str, default="./data/BTCV", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--msd_task", type=int, choices=[2, 9], default=2, help="MSD task number (2 for Heart, 9 for Spleen).")
    parser.add_argument("--crop_samples", type=int, default=1, help="Number of samples for random cropping.")

    #model 
    parser.add_argument("--model_name", type=str, required=True, default="unet3d", help="Model architecture to use.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of segmentation classes.")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels in the images.")
    parser.add_argument("--img_size", type=int, nargs=3, default=[96, 96, 96], help="Input image size (D, H, W) for the model.")
    parser.add_argument("--pretrain", type=str, default="", help="Path to pretrained model weights (optional).")

    #training
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision for training.")
    parser.add_argument("--scheduler_step", type=str, choices=["batch", "epoch"], default="epoch", help="When to step the learning rate scheduler.")  
    parser.add_argument("--T_max", type=int, default=100, help="T_max parameter for CosineAnnealingLR scheduler.")
    parser.add_argument("--eta_min", type=float, default=0.0, help="eta_min parameter for CosineAnnealingLR scheduler.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    
    #inference 
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint to load for inference.")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = load_args()
    print(args)





