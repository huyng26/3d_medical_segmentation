python -m medseg.training.train \
    --dataset       msd \
    --data_path     data/MSD \
    --model_name    swin_unetr \
    --num_classes   2 \
    --amp \
    --wandb_enable \
    --msd_task     9 \