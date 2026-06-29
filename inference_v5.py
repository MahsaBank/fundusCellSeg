import torch
from monai.data import DataLoader
from monai.data import Dataset as MonaiDataset
import zarr, json
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet, AttentionUnet
from monai.transforms import (
    Compose,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    ToTensord,
)
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset

def build_attUNet(in_channels=2, out_channels=2):
    return AttentionUnet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )

PH, PW = 192, 192

trfm = Compose([
    SpatialPadd(
        keys=["image", "label_cells", "label_focused"],
        spatial_size=(PH, PW),
        mode="constant"
    ),

    ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=5,
        upper=95,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    ToTensord(keys=["image", "label_cells", "label_focused"]),
])


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
            "key": k,
            "sub_key": sk,
            "z_key": zk
        }


@torch.no_grad()
def validate_model(
    checkpoint_filename,
    dataset_filename,
    save_filename,
    device="cpu",
    do_plot=True,
    save_probs=True,
    plot_slice=None,
    Twitch2b=False
):
    # Load checkpoint
    chk = torch.load(checkpoint_filename, map_location=device, weights_only=True)

    # Build network and Load dataset
    net = build_attUNet().to(device)
    net.load_state_dict(chk["network_state_dict"], strict=True)
    
    base_dataset = Zarr3DDataset(dataset_filename)

    if Twitch2b:
        with open("test_indices.json", "r") as f:
            test_indices = json.load(f)
    
        test_dataset = Subset(base_dataset, test_indices)
    else:
        test_dataset = base_dataset

    test_ds = MonaiDataset(test_dataset, transform=trfm)

    loader = DataLoader(test_ds, shuffle=False, batch_size=1)

    net.eval()

    zarr_f = zarr.open(save_filename, mode="a")

    for i, batch in enumerate(loader):
        x = batch["image"].to(device).float()
        l_c = batch["label_cells"].to(device).float()
        l_f = batch["label_focused"].to(device).float()
        logits = net(x)

        preds = logits
        if save_probs:
            preds = torch.sigmoid(logits)

        pred_focus = preds[:, 0:1] 
        pred_cells = preds[:, 1:2]

        x_np = x.detach().cpu().numpy()[0]      # (2, H, W) 
        y_c_np = pred_cells.detach().cpu().numpy()[0, 0]  # (H, W)
        l_c_np = l_c.detach().cpu().numpy()[0, 0]  # (H, W)
        y_f_np = pred_focus.detach().cpu().numpy()[0, 0]  # (H, W)
        l_f_np = l_f.detach().cpu().numpy()[0, 0]  # (H, W)
        pred2d_c_labels = (y_c_np > 0.8).astype(np.uint8)
        pred2d_f_labels = (y_f_np > 0.8).astype(np.uint8)

        key = batch["key"][0]
        sub_key = batch["sub_key"][0]
        z_key = batch["z_key"][0]
        slice_idx = int(z_key)

        # Save predictions
        case_group = zarr_f.require_group(str(key))
        sub_case_group = case_group.require_group(str(sub_key))
        slice_group = sub_case_group.require_group(str(slice_idx))
        slice_group.create_array("image", data=x_np, overwrite=True)
        slice_group.create_array("pred_cells_labels", data=pred2d_c_labels, overwrite=True)
        slice_group.create_array("pred_cells_probs", data=y_c_np, overwrite=True)
        slice_group.create_array("pred_focused_labels", data=pred2d_f_labels, overwrite=True)
        slice_group.create_array("pred_focused_probs", data=y_f_np, overwrite=True)

        # Plot predictions
        if do_plot:
            img = x_np
            pred2d_c = (y_c_np > 0.8).astype(np.uint8)
            pred2d_c = y_c_np

            pred2d_f = (y_f_np > 0.8).astype(np.uint8)
            pred2d_f = y_f_np

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # --- Frame 0 : Cells GT ---
            ax[0, 0].imshow(img[0], cmap="gray")
            ax[0, 0].imshow(l_c_np, cmap="jet", alpha=0.3)
            ax[0, 0].set_title(
                f"Frame 0 | Cells GT\nCase {key}/{sub_key} | Slice {slice_idx}"
            )
            ax[0, 0].axis("off")

            # --- Frame 0 : Cells Prediction ---
            ax[0, 1].imshow(img[0], cmap="gray")
            ax[0, 1].imshow(pred2d_c, cmap="jet", alpha=0.3)
            ax[0, 1].set_title("Frame 0 | Cells Prediction")
            ax[0, 1].axis("off")

            # --- Frame 1 : Focus GT ---
            ax[1, 0].imshow(img[1], cmap="gray")
            ax[1, 0].imshow(l_f_np, cmap="jet", alpha=0.3)
            ax[1, 0].set_title("Frame 1 | Focus GT")
            ax[1, 0].axis("off")

            # --- Frame 1 : Focus Prediction ---
            ax[1, 1].imshow(img[1], cmap="gray")
            ax[1, 1].imshow(pred2d_f, cmap="jet", alpha=0.3)
            ax[1, 1].set_title("Frame 1 | Focus Prediction")
            ax[1, 1].axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    checkpoint_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\train_attUNet\best_model.pth"
    dataset_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\raw data\Full Annotations--New ROIs\Ateam.zarr"
    save_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\test_Ateam_segmentation.zarr"

    validate_model(
        checkpoint_filename,
        dataset_filename,
        save_filename,
        save_probs=True,
        do_plot=False,
    )