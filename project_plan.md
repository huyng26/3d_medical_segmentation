# Project Execution Plan
*3D Medical Image Segmentation — 6-Week Roadmap*

---

## Overview

| Phase | Focus | Duration |
|-------|-------|----------|
| Phase 1 | Dataset Preparation & Preprocessing | Week 1 |
| Phase 2 | DataLoader & Augmentation Pipeline | Week 2 |
| Phase 3 | Model Implementation | Week 3 |
| Phase 4 | Training Loop & Experiment Tracking | Week 4 |
| Phase 5 | Inference & Evaluation | Week 5 |
| Phase 6 | Gradio Demo & Final Report | Week 6 |

---

## Phase 1 — Dataset Preparation & Preprocessing *(Week 1)*

**Goal:** Get both datasets downloaded, understood, and preprocessed into a training-ready format.

### BTCV
- Download from Synapse platform (requires free registration)
- Explore NIfTI files — check voxel spacing, intensity range, and label map across all 13 organ classes
- Apply CT preprocessing: clip HU values to [-175, 250], then z-score normalize
- Resample all volumes to a uniform voxel spacing (e.g. 1.5 × 1.5 × 2.0 mm)

### MSD Subset (Task03 Liver + Task09 Spleen)
- Download Task03 and Task09 only from medicaldecathlon.com (~6 GB total)
- Apply the same CT normalization and resampling pipeline as BTCV
- Verify label consistency: Task03 has background / liver / tumor, Task09 has background / spleen

### Deliverable
- Preprocessed volumes saved as `.npy` or `.nii.gz` in a structured folder per dataset
- A short data exploration notebook with histograms, slice previews, and label distribution plots

---

## Phase 2 — DataLoader & Augmentation Pipeline *(Week 2)*

**Goal:** Build a robust, reusable data pipeline that feeds all three models without modification.

### Dataset Class
- Implement a PyTorch `Dataset` class that loads preprocessed volumes and extracts random 3D patches (e.g. 128 × 128 × 64) at training time
- Support both BTCV and MSD with a unified interface — only the label map and file paths differ
- Implement sliding-window patch extraction for inference (no random sampling)

### Augmentation
- Use MONAI's augmentation library (strongly recommended for medical imaging)
- Apply: random flips (all axes), random rotation ±15°, random intensity scaling, Gaussian noise
- Keep augmentation light — medical data is scarce and aggressive augmentation can hurt performance

### Deliverable
- A DataLoader that yields `(image_patch, label_patch)` tensors ready for training
- A sanity check script that visualizes a batch of patches with overlaid masks

---

## Phase 3 — Model Implementation *(Week 3)*

**Goal:** Implement all three architectures cleanly and verify they run end-to-end on a dummy input.

### 3D U-Net
- Encoder-decoder with 3D convolutions, batch norm, and ReLU
- Skip connections from each encoder level to the corresponding decoder level
- Trilinear upsampling or transposed convolutions in the decoder

### SkipDenseNet3D
- Replace encoder blocks with 3D dense blocks (each layer concatenates all previous feature maps)
- Add inter-level skip connections between dense blocks across the encoder-decoder path
- Use bottleneck layers (1×1×1 conv) before each dense layer to control channel growth

### Swin UNETR
- Use the pretrained Swin Transformer encoder from MONAI (`monai.networks.nets.SwinUNETR`) — reimplementing the full transformer from scratch is impractical in the given timeline
- Connect the Swin encoder outputs to a CNN decoder via skip connections
- Fine-tune on your datasets from the publicly available pretrained weights

### Deliverable
- Each model passes a forward pass on a `(1, 1, 128, 128, 64)` dummy tensor without errors
- Parameter counts logged for all three models

---

## Phase 4 — Training Loop & Experiment Tracking *(Week 4)*

**Goal:** Train all three models on both datasets and track experiments systematically.

### Training Setup
- Loss: combined Dice loss + cross-entropy (50/50 weight)
- Optimizer: AdamW with initial lr = 1e-4 and cosine annealing scheduler
- Mixed precision: use `torch.cuda.amp.GradScaler` to fit within 12 GB VRAM
- Gradient checkpointing for Swin UNETR if memory is tight

### Experiment Tracking
- Use Weights & Biases (wandb) or TensorBoard to log train/val loss and DSC per epoch
- Save best checkpoint based on validation DSC
- Train 3D U-Net first — it converges fastest and validates your pipeline before moving to heavier models

### Deliverable
- Trained checkpoints for all three models on BTCV and MSD tasks
- Loss and DSC curves for each experiment

---

## Phase 5 — Inference & Evaluation *(Week 5)*

**Goal:** Run full-volume inference and compute final benchmark metrics.

### Inference
- Implement sliding-window inference: extract overlapping patches, run the model, and stitch predictions back using Gaussian importance weighting at patch borders (MONAI's `sliding_window_inference` handles this)
- Post-process predictions: apply argmax across class channels, optionally run connected component analysis to remove small spurious regions

### Evaluation
- Compute DSC, HD95, and IoU per class for every test volume
- Report mean and standard deviation across the test set
- Produce qualitative visualizations: overlay predicted masks on axial/sagittal/coronal slices for at least 3 representative cases per model

### Deliverable
- A results table comparing all three models across BTCV and MSD tasks
- Qualitative segmentation visualizations saved as figures

---

## Phase 6 — Gradio Demo & Final Report *(Week 6)*

**Goal:** Package the work into a presentable demo and write up findings.

### Gradio Demo
- Build a simple Gradio interface: upload a NIfTI file → select model → select axis → view segmentation overlay slice-by-slice
- Load model checkpoints at startup to avoid reload latency during the demo
- Run inference on GPU if available, CPU as fallback — for demo purposes, use a small cropped volume to keep response time under 5 seconds

### Final Report
- Summarize architecture decisions, training choices, and results
- Include the comparison table from Phase 5 and selected qualitative figures
- Discuss failure cases — where does each model struggle and why?

### Deliverable
- A working Gradio demo runnable with `python app.py`
- Final written report and cleaned-up codebase pushed to a GitHub repository
