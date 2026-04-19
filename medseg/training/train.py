"""Training loop with AMP, cosine LR scheduling, and checkpoint saving (Phase 4)."""
from __future__ import annotations

import argparse
import torch
from monai.losses.dice import DiceLoss
from monai.metrics.meandice import DiceMetric
from medseg.data_utils.btcv import build_btcv_dataloader
from medseg.data_utils.msd import build_msd_dataloader
from medseg.models import build_model
from medseg.cfg import load_args
from monai.inferers.utils import sliding_window_inference
from pathlib import Path
from tqdm import tqdm
from monai.data.utils import decollate_batch
from monai.transforms.post.array import AsDiscrete

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    wandb = None


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


def _checkpoint_path(args: argparse.Namespace, save_dir: Path) -> Path:
    if args.dataset == "msd":
        return save_dir / f"{args.model_name}_msd_task{args.msd_task}_best_model.pth"
    return save_dir / f"{args.model_name}_{args.dataset}_best_model.pth"

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, args:argparse.Namespace, scheduler=None) -> dict:
    """Run a single training epoch with mixed-precision gradients.

    Returns:
        Dict with keys ``loss`` (mean epoch loss).
    """
    model.train()
    total_loss = 0.0
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

def validation(
    epoch_iterator_val,
    model,
    loss_fn,
    device,
    args: argparse.Namespace,
    dice_metric: DiceMetric,
    global_step: int,
    max_iterations: int,
) -> dict:
    """Validation loop adapted to step-based training."""
    model.eval()
    total_loss = 0.0
    n = 0
    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    with torch.no_grad():
        for batch_data in epoch_iterator_val:
            val_inputs = batch_data["image"].to(device)
            val_labels = batch_data["label"].to(device)
            with torch.autocast("cuda", enabled=args.amp):
                val_outputs = sliding_window_inference(
                    val_inputs,
                    tuple(args.img_size),
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.25,
                )
                loss = loss_fn(val_outputs, val_labels)

            if not isinstance(val_outputs, torch.Tensor):
                if isinstance(val_outputs, tuple):
                    val_outputs = val_outputs[0]
                elif isinstance(val_outputs, dict):
                    val_outputs = next(iter(val_outputs.values()))
                else:
                    raise TypeError(f"Unsupported model output type: {type(val_outputs)}")

            total_loss += float(loss.detach().item())
            n += 1

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert: list[torch.Tensor] = []
            for val_label_tensor in val_labels_list:
                label_post = post_label(val_label_tensor)
                if isinstance(label_post, torch.Tensor):
                    val_labels_convert.append(label_post)
                else:
                    val_labels_convert.append(torch.as_tensor(label_post, device=device))

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert: list[torch.Tensor] = []
            for val_pred_tensor in val_outputs_list:
                pred_post = post_pred(val_pred_tensor)
                if isinstance(pred_post, torch.Tensor):
                    val_output_convert.append(pred_post)
                else:
                    val_output_convert.append(torch.as_tensor(pred_post, device=device))
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(f"Validate ({global_step} / {float(max_iterations):.1f} Steps)")

    metric_agg = dice_metric.aggregate()
    if isinstance(metric_agg, tuple):
        metric_agg = metric_agg[0]
    mean_dice_val = float(metric_agg.item())
    dice_metric.reset()
    return {"dsc_mean": mean_dice_val, "loss": total_loss / max(1, n)}


def train(
    global_step: int,
    train_loader,
    val_loader,
    model,
    loss_function,
    optimizer,
    scaler,
    scheduler,
    device,
    args: argparse.Namespace,
    dice_val_best: float,
    global_step_best: int,
    save_dir: Path,
    wandb_run=None,
):
    """Step-based training loop with periodic validation and best-checkpoint saving."""
    model.train()
    epoch_loss = 0.0
    step = 0
    epoch_loss_values = []
    metric_values = []
    max_iterations = args.max_iterations
    eval_num = max(1, args.eval_num)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch_data in enumerate(epoch_iterator):
        if global_step >= max_iterations:
            break

        step += 1
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device)

        with torch.autocast("cuda", enabled=args.amp):
            logits = model(image)
            loss = loss_function(logits, label)

        scaler.scale(loss).backward()
        epoch_loss += float(loss.detach().item())
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()

        epoch_iterator.set_description(
            f"Training ({global_step} / {max_iterations} Steps) (loss={float(loss.detach().item()):2.5f})"
        )

        should_eval = ((global_step % eval_num == 0 and global_step != 0) or (global_step == max_iterations - 1))
        if should_eval:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            val_metrics = validation(
                epoch_iterator_val,
                model,
                loss_function,
                device,
                args,
                dice_metric,
                global_step,
                max_iterations,
            )

            epoch_loss = epoch_loss / max(1, step)
            epoch_loss_values.append(epoch_loss)
            metric_values.append(val_metrics["dsc_mean"])

            if val_metrics["dsc_mean"] > dice_val_best:
                dice_val_best = val_metrics["dsc_mean"]
                global_step_best = global_step
                torch.save(model.state_dict(), _checkpoint_path(args, save_dir))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best,
                        val_metrics["dsc_mean"],
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best,
                        val_metrics["dsc_mean"],
                    )
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": epoch_loss,
                        "val/loss": val_metrics["loss"],
                        "val/dice": val_metrics["dsc_mean"],
                    },
                    step=global_step + 1,
                )

            model.train()
            epoch_loss = 0.0
            step = 0

        global_step += 1

    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()

    return global_step, dice_val_best, global_step_best


def main(args: argparse.Namespace) -> None:
    """Entry point: parse args, build components, and run the training loop."""
    model = build_model(args.model_name, cfg=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, test_loader = _build_loaders(args)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    scaler = torch.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb_enabled:
        if wandb is None:
            print("wandb is not installed. Skipping wandb logging.")
        else:
            wandb_run = wandb.init(
                project="3d-medical-segmentation",
                config=vars(args),
                dir=str(save_dir),
            )

    max_iterations = args.max_iterations
    if max_iterations <= 0:
        train_steps = len(train_loader) if train_loader is not None else 0
        max_iterations = args.num_epochs * max(1, train_steps)
        args.max_iterations = max_iterations

    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0

    while global_step < args.max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step,
            train_loader,
            test_loader,
            model,
            loss_fn,
            optimizer,
            scaler,
            scheduler,
            device,
            args,
            dice_val_best,
            global_step_best,
            save_dir,
            wandb_run,
        )

    print(
        f"Training complete. Best Dice={dice_val_best:.4f} at global_step={global_step_best}"
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = load_args()
    main(args)
