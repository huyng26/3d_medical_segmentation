python -m medseg.training.train \
    --dataset       btcv \
    --data_path     data/BTCV \
    --model_name    swin_unetr \
    --num_classes   14 \
    --amp \
    --wandb_enable