"""Training loop with AMP, cosine LR scheduling, and checkpoint saving (Phase 4)."""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from medseg.data_utils.btcv import build_btcv_dataloader
from medseg.data_utils.msd import build_msd_dataloader
from medseg.models import build_model
from medseg.cfg import load_args
from monai.inferers import sliding_window_inference 
from pathlib import Path
from tqdm import tqdm


def build_scheduler(optimizer, args: argparse.Namespace):
    """Return a cosine-annealing LR scheduler configured from ``cfg``."""
    t_max = args.T_max
    eta_min = args.eta_min
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max),
        eta_min=eta_min,
    )
    return scheduler

def _build_loaders(args: argparse.Namespace):
    if args.dataset == "btcv":
        train_loader = build_btcv_dataloader(args, mode="train")
        test_loader = build_btcv_dataloader(args, mode="test")
    elif args.dataset == "msd":
        train_loader = build_msd_dataloader(args, mode="train", task=args.msd_task)
        test_loader = build_msd_dataloader(args, mode="test", task=args.msd_task)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, args:argparse.Namespace, scheduler=None) -> dict:
    """Run a single training epoch with mixed-precision gradients.

    Returns:
        Dict with keys ``loss`` (mean epoch loss).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    scheduler_step = args.scheduler_step
    use_amp = args.amp
    n =  0
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
        n += 1

    if scheduler is not None and scheduler_step != "batch":
        scheduler.step()
    
    return {"loss": total_loss / max(1, n)}

def validate(model, loader, loss_fn, device, num_classes: int, args: argparse.Namespace) -> dict:
    """Evaluate on the validation set using sliding-window inference.

    Returns:
        Dict with keys ``loss`` and ``dsc_mean``.
    """
    model.eval()
    total_loss = 0.0
    dsc_metric = DiceMetric(include_background=False, reduction="mean")
    n = 0
    with torch.no_grad():
        for batch_data in loader: 
            image, label = batch_data["image"].to(device), batch_data["label"].to(device)
            with torch.autocast("cuda", enabled=args.amp):
                roi_size = tuple(args.img_size)
                logits = sliding_window_inference(image, roi_size,sw_batch_size=4,predictor=model, overlap=0.25)
                loss = loss_fn(logits, label)
            total_loss += float(loss.detach().item())
            n += 1
            pred = torch.argmax(logits, dim=1)  # [B, D, H, W]
            pred_1h = F.one_hot(pred.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

            if label.ndim == 5 and label.shape[1] == 1:
                labels_idx = label[:, 0].long()
            elif label.ndim == 4:
                labels_idx = label.long()
            else:
                raise ValueError(f"Unexpected label shape: {tuple(label.shape)}")

            labels_1h = F.one_hot(labels_idx, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            dsc_metric(y_pred=pred_1h, y=labels_1h)

    dsc_mean = float(dsc_metric.aggregate().item())
    dsc_metric.reset()

    return {"loss": total_loss / max(1, n), "dsc_mean": dsc_mean}


def main(args: argparse.Namespace) -> None:
    """Entry point: parse args, build components, and run the training loop."""
    model = build_model(args.model_name, cfg=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, test_loader= _build_loaders(args)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    scaler = torch.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    num_epochs = args.num_epochs
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dsc = 0.0 

    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        train_metrics = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            loss_fn, 
            device, 
            args, 
            scheduler
        )

        test_metrics = validate(
            model,
            test_loader,
            loss_fn,
            device,
            args.num_classes,
            args
        )

        print(
            f"Epoch [{epoch:03d}/{args.num_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={test_metrics['loss']:.4f} | "
            f"DSC={test_metrics['dsc_mean']:.4f}"
        )

        if test_metrics["dsc_mean"] > best_dsc:
            best_dsc = test_metrics["dsc_mean"]
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"Saved best model with DSC = {best_dsc:.4f}")


if __name__ == "__main__":
    args = load_args()
    main(args)
