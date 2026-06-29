import numpy as np
import zarr, json
import torch
import torch.nn.functional as F
from monai.data import list_data_collate, DataLoader
from scipy.ndimage import distance_transform_edt
from monai.transforms import (
    Compose, SpatialPadd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandFlipd, RandGaussianNoised, ToTensord
)
from monai.networks.nets import UNet, AttentionUnet
from torch.optim import AdamW
import os
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

from torch.utils.data import random_split, Subset
from monai.data import Dataset as MonaiDataset
from torch.utils.data import Dataset as TorchDataset


def focal_dice_loss(logits, target, alpha=0.9, gamma=2.0, dice_w=0.3, eps=1e-6):
    target = target.float()

    # --- Focal BCE (per-pixel) ---
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)

    pt = p * target + (1 - p) * (1 - target)                 # pt per pixel
    alpha_t = alpha * target + (1 - alpha) * (1 - target)    # class-balanced alpha

    focal = (alpha_t * (1 - pt).pow(gamma) * bce).mean()

    # --- Soft Dice ---
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims) + eps
    dice = 1 - ((2 * inter + eps) / denom)
    dice = dice.mean()

    return (1 - dice_w) * focal + dice_w * dice

def build_smallUNet(in_channels=1, out_channels=1):
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(16, 32, 64),
        strides=(1, 1),
        num_res_units=0,
    )

def build_attUNet(in_channels=2, out_channels=2):
    return AttentionUnet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )

class Zarr3DDataset(TorchDataset):
    def __init__(self, zarr_path, min_cell_pixels=0):
        print(zarr_path)

        self.zarr_file = zarr.open(zarr_path, mode="r")
        self.index_map = []

        for k in self.zarr_file.keys():
            for sk in self.zarr_file[k].keys():
                z_stack_grp = self.zarr_file[k][sk]["z_stacks"]

                for zk in z_stack_grp.keys():
                    grp = z_stack_grp[zk]

                    mask_cells = np.asarray(grp["cells_roi"])  # (P, H, W)

                    # union of frame 0 and frame 1 masks
                    mask2d_cells = (mask_cells[0] + mask_cells[1]) > 0

                    if mask2d_cells.sum() > min_cell_pixels:
                        self.index_map.append((k, sk, zk))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        k, sk, zk = self.index_map[idx]
        grp = self.zarr_file[k][sk]["z_stacks"][zk]

        # raw frames: (P, H, W)
        raw = np.asarray(grp["raw"]).astype(np.float32)

        # use frame 0 and frame 1 as 2 input channels
        image = raw[:2]  # (2, H, W)

        # cell mask union
        mask_cells = np.asarray(grp["cells_roi"])
        label_cells = ((mask_cells[0] + mask_cells[1]) > 0).astype(np.float32)
        label_cells = np.expand_dims(label_cells, axis=0)  # (1, H, W)

        # focused mask intersection: area focused in both frames
        mask_focus = np.asarray(grp["not_out_of_focus_roi"])
        label_focused = ((mask_focus[0] > 0) & (mask_focus[1] > 0)).astype(np.float32)
        label_focused = np.expand_dims(label_focused, axis=0)  # (1, H, W)

        return {
            "image": image,
            "label_cells": label_cells,
            "label_focused": label_focused,
        }
    
PH, PW = 192, 192

train_trfm = Compose([
    SpatialPadd(
        keys=["image", "label_cells", "label_focused"],
        spatial_size=(PH, PW),
        mode="constant"
    ),

    ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label_cells", "label_focused"],
        label_key="label_cells",
        spatial_size=(PH, PW),
        pos=0.5,
        neg=0.5,
        num_samples=4,
    ),

    ToTensord(keys=["image", "label_cells", "label_focused"]),
])

val_trfm = Compose([
    SpatialPadd(
        keys=["image", "label_cells", "label_focused"],
        spatial_size=(PH, PW),
        mode="constant"
    ),

    ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    ToTensord(keys=["image", "label_cells", "label_focused"]),
])

def bce_dice_focal_loss(logits, target, alpha=0.75, gamma=2.0, eps=1e-6, w_focal=0.5):
    prob = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    p_t = prob * target + (1 - prob) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = (focal_weight * bce).mean()

    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims) + eps
    dice_loss = 1.0 - (2.0 * inter + eps) / denom
    dice_loss = dice_loss.mean()

    return w_focal * focal_loss + (1 - w_focal) * dice_loss

def train_UNet(
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    device: str = "cuda",
    max_epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    log_every: int = 20,
    checkpoint_path: str = "",
    available_checkpoint=None,
    total_w_loss_focus=0.3
):
    os.makedirs(checkpoint_path, exist_ok=True)

    bestmodel_filename = os.path.join(checkpoint_path, "best_model.pth")
    csv_file = os.path.join(checkpoint_path, "metrics.csv")

    # Create CSV header only if file does not exist
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "global_step",
                "train_loss_avg",
                "train_cells_loss_avg",
                "train_focus_loss_avg",
                "val_loss_avg",
                "val_cells_loss_avg",
                "val_focus_loss_avg",
                "label_cells_frac_avg",
                "label_focus_frac_avg",
                "lr",
            ])

    model = build_attUNet().to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        patience=8,
        factor=0.5,
        min_lr=1e-6,
        threshold=1e-4,
    )

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_score = float("inf")

    # Resume checkpoint if provided
    if available_checkpoint:
        chk = torch.load(available_checkpoint, map_location=device)

        model.load_state_dict(chk["network_state_dict"])

        if "optimizer_state_dict" in chk:
            opt.load_state_dict(chk["optimizer_state_dict"])

        start_epoch = chk.get("epoch", 0) + 1
        global_step = chk.get("global_step", 0)
        best_score = chk.get("best_score", float("inf"))

        print(f"Loaded checkpoint from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, max_epochs):
        model.train()

        run_loss = 0.0
        run_loss_cells = 0.0
        run_loss_focus = 0.0
        run_cells_frac = 0.0
        run_focus_frac = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y_c = batch["label_cells"].to(device)
            y_f = batch["label_focused"].to(device)

            if 1:
                fig, ax = plt.subplots(2, 1, figsize=(8, 12))
                ax[0].imshow(x[0, 0, :, :].cpu(), cmap="gray")
                ax[0].imshow(y_c[0, 0, :, :].cpu(), alpha=0.2)
                ax[0].imshow(y_f[0, 0, :, :].cpu(), alpha=0.1)
                ax[1].imshow(x[0, 1, :, :].cpu(), cmap="gray")
                ax[1].imshow(y_c[0, 0, :, :].cpu(), alpha=0.2)
                ax[1].imshow(y_f[0, 0, :, :].cpu(), alpha=0.1)

                plt.show()

            cells_frac = (y_c.sum() / (y_c.numel() + 1e-6)).item()
            focus_frac = (y_f.sum() / (y_f.numel() + 1e-6)).item()

            run_cells_frac += cells_frac
            run_focus_frac += focus_frac

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x)  # (B, 2, H, W)

                    pred_focus = logits[:, 0:1]
                    pred_cells = logits[:, 1:2]

                    loss_focus = bce_dice_focal_loss(
                        pred_focus,
                        y_f,
                        w_focal=0.5,
                    )

                    loss_cells = bce_dice_focal_loss(
                        pred_cells,
                        y_c,
                        w_focal=0.7,
                    )

                    loss = loss_cells + total_w_loss_focus * loss_focus

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            else:
                logits = model(x)  # (B, 2, H, W)

                pred_focus = logits[:, 0:1]
                pred_cells = logits[:, 1:2]

                loss_focus = bce_dice_focal_loss(
                    pred_focus,
                    y_f,
                    w_focal=0.5,
                )

                loss_cells = bce_dice_focal_loss(
                    pred_cells,
                    y_c,
                    w_focal=0.7,
                )

                loss = loss_cells + total_w_loss_focus * loss_focus

                loss.backward()
                opt.step()

            run_loss += loss.item()
            run_loss_cells += loss_cells.item()
            run_loss_focus += loss_focus.item()
            n_steps += 1

            if (step % log_every) == 0:
                print(
                    f"epoch {epoch:03d} step {step:04d} | "
                    f"loss={loss.item():.4f} | "
                    f"cells={loss_cells.item():.4f} | "
                    f"focus={loss_focus.item():.4f} | "
                    f"cells_frac={cells_frac:.4f} | "
                    f"focus_frac={focus_frac:.4f}"
                )

            global_step += 1

        train_loss_avg = run_loss / max(n_steps, 1)
        train_cells_loss_avg = run_loss_cells / max(n_steps, 1)
        train_focus_loss_avg = run_loss_focus / max(n_steps, 1)
        cells_frac_avg = run_cells_frac / max(n_steps, 1)
        focus_frac_avg = run_focus_frac / max(n_steps, 1)

        val_loss_avg = None
        val_cells_loss_avg = None
        val_focus_loss_avg = None

        if val_loader is not None:
            model.eval()

            vloss = 0.0
            vloss_cells = 0.0
            vloss_focus = 0.0
            vsteps = 0

            with torch.no_grad():
                for vb in val_loader:
                    vx = vb["image"].to(device)
                    vy_c = vb["label_cells"].to(device)
                    vy_f = vb["label_focused"].to(device)

                    vlogits = model(vx)

                    v_pred_focus = vlogits[:, 0:1]
                    v_pred_cells = vlogits[:, 1:2]

                    v_loss_focus = bce_dice_focal_loss(
                        v_pred_focus,
                        vy_f,
                        w_focal=0.5,
                    )

                    v_loss_cells = bce_dice_focal_loss(
                        v_pred_cells,
                        vy_c,
                        w_focal=0.7,
                    )

                    v_loss = v_loss_cells + total_w_loss_focus * v_loss_focus

                    vloss += v_loss.item()
                    vloss_cells += v_loss_cells.item()
                    vloss_focus += v_loss_focus.item()
                    vsteps += 1

            val_loss_avg = vloss / max(vsteps, 1)
            val_cells_loss_avg = vloss_cells / max(vsteps, 1)
            val_focus_loss_avg = vloss_focus / max(vsteps, 1)

            print(
                f"[VAL] epoch {epoch:03d} | "
                f"loss={val_loss_avg:.4f} | "
                f"cells={val_cells_loss_avg:.4f} | "
                f"focus={val_focus_loss_avg:.4f}"
            )

        if val_loss_avg is not None:
            scheduler.step(val_loss_avg)
        else:
            scheduler.step(train_loss_avg)

        current_lr = opt.param_groups[0]["lr"]
        print(f"[LR] epoch {epoch:03d} lr={current_lr:.2e}")

        score = val_loss_avg if val_loss_avg is not None else train_loss_avg

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "network_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "best_score": best_score,
            "train_loss_avg": train_loss_avg,
            "val_loss_avg": val_loss_avg,
        }

        torch.save(
            checkpoint,
            os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch:03d}.pth"),
        )

        if score < best_score:
            best_score = score
            checkpoint["best_score"] = best_score

            torch.save(checkpoint, bestmodel_filename)

            print(
                f"[BEST] epoch {epoch:03d} "
                f"score={best_score:.4f} -> saved best_model.pth"
            )

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                global_step,
                train_loss_avg,
                train_cells_loss_avg,
                train_focus_loss_avg,
                val_loss_avg if val_loss_avg is not None else "",
                val_cells_loss_avg if val_cells_loss_avg is not None else "",
                val_focus_loss_avg if val_focus_loss_avg is not None else "",
                cells_frac_avg,
                focus_frac_avg,
                current_lr,
            ])

        print(
            f"[TRAIN] epoch {epoch:03d} avg | "
            f"loss={train_loss_avg:.4f} | "
            f"cells={train_cells_loss_avg:.4f} | "
            f"focus={train_focus_loss_avg:.4f} | "
            f"cells_frac={cells_frac_avg:.4f} | "
            f"focus_frac={focus_frac_avg:.4f}"
        )

    return model

dataset_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\raw data\Full Annotations--New ROIs\Twitch2b.zarr"
dataset_filename = "/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/raw data/Full Annotations--New ROIs/Twitch2b.zarr" 

base_dataset = Zarr3DDataset(dataset_filename)

# Load saved test indices
with open("test_indices.json", "r") as f:
    test_indices = json.load(f)

test_indices = set(test_indices)

# Keep everything except the test set
train_val_indices = [
    i for i in range(len(base_dataset))
    if i not in test_indices
]

train_val_dataset = Subset(base_dataset, train_val_indices)

# Split remaining data into train/validation
train_size = int(0.75 * len(train_val_dataset))  # 75% of remaining = 60% overall if test is 20%
val_size = len(train_val_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_base, val_base = random_split(
    train_val_dataset,
    [train_size, val_size],
    generator=generator
)

# n = len(base_dataset)

# train_size = int(0.6 * n)
# remaining = n - train_size

# val_size = remaining // 2
# test_size = remaining - val_size

# train_base, val_base, test_base = random_split(
#     base_dataset,
#     [train_size, val_size, test_size],
#     generator=generator
# )

# with open("test_indices.json", "w") as f:
#     json.dump(test_base.indices, f)

train_dataset = MonaiDataset(train_base, transform=train_trfm)
val_dataset = MonaiDataset(val_base, transform=val_trfm) 

loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=list_data_collate
)

loader_val = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=list_data_collate
)

model = train_UNet(
    train_loader=loader,
    val_loader=loader_val,
    device="cpu",
    max_epochs=1600,
    checkpoint_path = "/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/train_attUNet", # r"Z:\bmahsa\workplaces\CellFundusSegmentation\train_attUNet",
    available_checkpoint = "/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/train_attUNet/best_model.pth", #r"Z:\bmahsa\workplaces\CellFundusSegmentation\train_attUNet\best_model.pth"
    total_w_loss_focus=0.1
)