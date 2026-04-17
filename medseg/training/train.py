"""Training loop with AMP, cosine LR scheduling, and checkpoint saving (Phase 4)."""
from __future__ import annotations

import argparse
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from medseg.data_utils.btcv import build_btcv_dataloader
from medseg.data_utils.msd import build_msd_dataloader
from medseg.models import build_model
from medseg.utils.config import load_config
from monai.inferers import sliding_window_inference 

def build_scheduler(optimizer, cfg: dict):
    """Return a cosine-annealing LR scheduler configured from ``cfg``."""
    scheduler_cfg = cfg.get("scheduler", {})
    training_cfg = cfg.get("training", {})
    t_max = int(scheduler_cfg.get("T_max", training_cfg.get("max_epochs", cfg.get("num_epochs", 1))))
    eta_min = float(scheduler_cfg.get("eta_min", cfg.get("min_lr", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max),
        eta_min=eta_min,
    )
    return scheduler

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg: dict, scheduler=None) -> dict:
    """Run a single training epoch with mixed-precision gradients.

    Returns:
        Dict with keys ``loss`` (mean epoch loss).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    scheduler_step = str(cfg.get("scheduler_step", "epoch")).lower()
    use_amp = bool(cfg.get("training", {}).get("amp", True)) and device.type == "cuda"
    len_loader = len(loader)

    for batch_data in loader:
        image = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(image)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scheduler_step == "batch":
            scheduler.step()

        total_loss += float(loss.detach().item())

    if scheduler is not None and scheduler_step != "batch":
        scheduler.step()
    
    total_loss = total_loss / len_loader if len_loader > 0 else 0.0
    return {"loss": total_loss}

def validate(model, loader, loss_fn, device, num_classes: int, cfg: dict) -> dict:
    """Evaluate on the validation set using sliding-window inference.

    Returns:
        Dict with keys ``loss`` and ``dsc_mean``.
    """
    model.eval()
    total_loss = 0.0
    dsc_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, num_classes=cfg.get("num_classes", num_classes))
    with torch.no_grad():
        for batch_data in loader: 
            image, label = batch_data["image"].to(device), batch_data["label"].to(device)
            with torch.autocast("cuda"):
                roi_size = cfg.get("roi_size", [96, 96, 96])
                overlap = float(cfg.get("overlap", 0.25))
                outputs = sliding_window_inference(
                    image,
                    roi_size,
                    4,
                    model,
                )
                loss = loss_fn(outputs, label)
                total_loss += float(loss.detach().item())

def main(args: argparse.Namespace | None = None) -> None:
    """Entry point: parse args, build components, and run the training loop."""
    model = build_model(args.model_name, args.num_classes, cfg=args)
    train_loader = build_btcv_dataloader(args, mode="train")
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    scaler = torch.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, cfg=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = args.num_epochs


if __name__ == "__main__":
    cfg = load_config("D:\Code\3d_medical_segmentation\configs\train_default.yaml")
    print(cfg)
    args = argparse.Namespace(**cfg)
