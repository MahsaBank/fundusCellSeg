import os
import zipfile
import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from monai.transforms import (
    Compose,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from monai.networks.nets import AttentionUnet
from tifffile import imread
import glob
from torch.utils.data import Dataset as TorchDataset
from monai.data import DataLoader
from monai.data import Dataset as MonaiDataset
from roifile import ImagejRoi, ROI_TYPE
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _cosine(a, b, eps=1e-8):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def postprocess_probs(
    prob_volume,
    threshold=0.8,
    min_size=20,
    use_watershed=True,
    min_distance=6,
):
    """
    prob_volume: shape (Z, H, W)
    returns:
        cleaned: binary mask, shape (Z, H, W)
        labels_2d: labeled instances per z-slice, shape (Z, H, W)
    """

    binary_volume = prob_volume > threshold

    cleaned = np.zeros_like(binary_volume, dtype=bool)
    labels_2d = np.zeros_like(binary_volume, dtype=np.int32)

    for z in range(binary_volume.shape[0]):
        clean_z = remove_small_objects(
            binary_volume[z].astype(bool),
            min_size=min_size,
            connectivity=1,
        )

        cleaned[z] = clean_z

        if not np.any(clean_z):
            continue

        if use_watershed:
            distance = ndi.distance_transform_edt(clean_z)

            coords = peak_local_max(
                distance,
                min_distance=min_distance,
                labels=clean_z,
            )

            markers = np.zeros_like(clean_z, dtype=np.int32)

            for i, (r, c) in enumerate(coords, start=1):
                markers[r, c] = i

            if markers.max() > 0:
                labels_z = watershed(
                    -distance,
                    markers,
                    mask=clean_z,
                )
            else:
                labels_z = label(clean_z, connectivity=1)

        else:
            labels_z = label(clean_z, connectivity=1)

        labels_2d[z] = labels_z.astype(np.int32)

    return cleaned, labels_2d


def save_ellipse_rois_to_zip(cells, output_zip):
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)

    seen = set()

    with zipfile.ZipFile(output_zip, "w") as zf:
        for c in cells:
            top = int(round(c["top"]))
            left = int(round(c["left"]))
            h = int(round(c["height"]))
            w = int(round(c["width"]))

            z = int(c["z"])
            cell_id = int(c["cell_id"])
            roi_name = f"Cell_{cell_id:04d}_z{z+1:03d}"

            if roi_name in seen:
                continue

            seen.add(roi_name)
            
            roi = ImagejRoi(
                roitype=ROI_TYPE.OVAL,
                top=top,
                left=left,
                bottom=top + h,
                right=left + w,
            )


            roi.position = 0
            roi.c_position = 0
            roi.z_position = z + 1
            roi.t_position = 0

            roi.name = roi_name
            
            zf.writestr(f"{roi.name}.roi", roi.tobytes())

class tiffDataset(TorchDataset):
    def __init__(self, tif_file, min_z=None, max_z=None):
        self.img = np.asarray(imread(tif_file)).astype(np.float32)

        # Expected original shape: (Z, P, H, W)
        # Convert to: (P, Z, H, W)
        self.img = self.img.transpose((1, 0, 2, 3))

        P, Z, H, W = self.img.shape

        min_z = 0 if min_z is None else min_z
        max_z = Z - 1 if max_z is None else max_z
        self.index = range(min_z, max_z + 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        z = self.index[idx]

        # raw frames: (P, H, W)
        raw = self.img[:, z, :, :]

        # use frame 0 and frame 1 as 2 input channels
        image = raw[:2]  # (2, H, W)

        return {
            "image": image,
        }

def components_to_cells(labels_2d):
    cells = []

    for z in range(labels_2d.shape[0]):
        for r in regionprops(labels_2d[z]):
            minr, minc, maxr, maxc = r.bbox

            cells.append({
                "top": float(minr),
                "left": float(minc),
                "height": float(maxr - minr),
                "width": float(maxc - minc),
                "position": {"slice": z},  # 0-based for your function
            })

    return cells


def _crop_patch(img2d, cy, cx, half):
    H, W = img2d.shape
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    patch = img2d[y0:y1, x0:x1]
    return patch

def _hist_feat(patch, bins=32, eps=1e-8):
    # robust histogram on patch intensities
    p = patch.astype(np.float32)
    vmin, vmax = float(np.min(p)), float(np.max(p))
    if vmax <= vmin + eps:
        # flat patch
        h = np.zeros((bins,), np.float32)
        h[0] = 1.0
        return h
    h, _ = np.histogram(p, bins=bins, range=(vmin, vmax))
    h = h.astype(np.float32)
    h /= (h.sum() + eps)
    return h

def roi_center_from_values(top, left, h, w):
    cy = float(top) + float(h) / 2.0
    cx = float(left) + float(w) / 2.0
    return cy, cx


def merge_cells_by_hist_to_rois(
    raw3d,
    cells,
    radius=2,
    patch_half=8,
    bins=16,
    sim_thr=0.92,
    mean_rel_thr=0.20,
    std_rel_thr=0.40,
    max_center_dist=6,
    eps=1e-8,
):
    """
    Merge predicted 2D cell ROIs across nearby z-slices using histogram similarity.

    raw3d: (Z, H, W)
    cells: list of dicts with:
        top, left, height, width, position["slice"]

    returns:
        merged_cells: list of ROI records
        One output ROI per merged cell track.
    """

    assert raw3d.ndim == 3, f"raw3d must be (Z,H,W), got {raw3d.shape}"

    Z, H, W = raw3d.shape
    raw = raw3d.astype(np.float32)

    # Precompute features for all predicted cells
    valid_cells = []

    for idx, c in enumerate(cells):
        z = int(c["position"]["slice"])

        if not (0 <= z < Z):
            continue

        top = float(c["top"])
        left = float(c["left"])
        hh = float(c["height"])
        ww = float(c["width"])

        if hh <= 0 or ww <= 0:
            continue

        cy, cx = roi_center_from_values(top, left, hh, ww)
        cy = int(round(cy))
        cx = int(round(cx))

        if not (0 <= cy < H and 0 <= cx < W):
            continue

        patch = _crop_patch(raw[z], cy, cx, patch_half)

        if patch.size == 0:
            continue

        valid_cells.append({
            "z": z,
            "top": top,
            "left": left,
            "height": hh,
            "width": ww,
            "cy": cy,
            "cx": cx,
            "hist": _hist_feat(patch, bins=bins),
            "mean": float(patch.mean()),
            "std": float(patch.std()),
        })

    n = len(valid_cells)

    if n == 0:
        return []

    # Union-find structure
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    # Compare only cells in nearby z-slices
    for i in range(n):
        c1 = valid_cells[i]

        for j in range(i + 1, n):
            c2 = valid_cells[j]

            dz = abs(c1["z"] - c2["z"])

            if dz == 0:
                continue

            if dz > radius:
                continue

            dy = c1["cy"] - c2["cy"]
            dx = c1["cx"] - c2["cx"]
            dist = (dy ** 2 + dx ** 2) ** 0.5

            if dist > max_center_dist:
                continue

            sim = _cosine(c1["hist"], c2["hist"])

            mean_ok = (
                abs(c2["mean"] - c1["mean"]) / (abs(c1["mean"]) + eps)
            ) <= mean_rel_thr

            std_ok = (
                abs(c2["std"] - c1["std"]) / (abs(c1["std"]) + eps)
            ) <= std_rel_thr

            if sim >= sim_thr and mean_ok and std_ok:
                union(i, j)

    # Group merged tracks
    groups = {}

    for i, c in enumerate(valid_cells):
        root = find(i)

        if root not in groups:
            groups[root] = []

        groups[root].append(c)

    merged_cells = []

    for new_cell_id, group in enumerate(groups.values(), start=1):
        group = sorted(group, key=lambda x: x["z"])

        for g in group:

            merged_cells.append({
                "cell_id": new_cell_id,
                "z": g["z"],
                "top": g["top"],
                "left": g["left"],
                "height": g["height"],
                "width": g["width"],
            })

    return merged_cells

def run_prediction_to_imagej_rois(
    input_dir,
    output_dir,
    chk_filename="best_model.pth",
    device="cpu",
    cell_threshold=0.8,
    min_size=20,
    focus_threshold=0.8,
):
    os.makedirs(output_dir, exist_ok=True)

    model = build_attUNet(in_channels=2, out_channels=2).to(device)

    chk = torch.load(chk_filename, map_location=device)
    model.load_state_dict(chk["network_state_dict"])
    model.eval()

    tiff_files = glob.glob(os.path.join(input_dir, "*.tif"))

    for tiff_file in tiff_files:
        print(f"Processing: {tiff_file}")

        name = os.path.splitext(os.path.basename(tiff_file))[0]

        dataset = tiffDataset(tiff_file)
        ds = MonaiDataset(dataset, transform=trfm)

        loader = DataLoader(
            ds,
            shuffle=False,
            batch_size=1,
        )

        cell_probs = []
        focus_probs = []

        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device).float()

                logits = model(x)
                probs = torch.sigmoid(logits)

                pred_focus = probs[:, 0:1]
                pred_cells = probs[:, 1:2]

                focus_probs.append(pred_focus[0, 0].cpu().numpy())
                cell_probs.append(pred_cells[0, 0].cpu().numpy())

        focus_volume = np.stack(focus_probs, axis=0)  # (Z, H, W)
        cell_volume = np.stack(cell_probs, axis=0)    # (Z, H, W)

        focus_mask = focus_volume > focus_threshold

        # Optional: uncomment if you want to suppress cells outside focused regions
        # cell_volume = cell_volume * focus_mask

        raw_volume = np.asarray(imread(tiff_file)).astype(np.float32)  # (Z, P, H, W)

        raw_volume_for_hist = np.maximum(
            raw_volume[:, 0, :, :],
            raw_volume[:, 1, :, :]
        )  # (Z, H, W)

        binary_volume, labels_2d = postprocess_probs(
            cell_volume,
            threshold=cell_threshold,
            min_size=min_size,
        )

        cells = components_to_cells(labels_2d)

        roi_cells = merge_cells_by_hist_to_rois(raw_volume_for_hist, cells)

        output_zip = os.path.join(
            output_dir,
            f"{name}_predicted_cells.zip",
        )

        save_ellipse_rois_to_zip(
            cells=roi_cells,
            output_zip=output_zip,
        )

        np.save(
            os.path.join(output_dir, f"{name}_cell_prob.npy"),
            cell_volume,
        )

        np.save(
            os.path.join(output_dir, f"{name}_focus_prob.npy"),
            focus_volume,
        )

        np.save(
            os.path.join(output_dir, f"{name}_binary.npy"),
            binary_volume.astype(np.uint8),
        )

        np.save(
            os.path.join(output_dir, f"{name}_labels.npy"),
            labels_2d,
        )

        print(f"Saved: {output_zip}")

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
        keys=["image"],
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

    ToTensord(keys=["image"]),
])


run_prediction_to_imagej_rois(
    chk_filename=r"Z:\bmahsa\workplaces\CellFundusSegmentation\train_attUNet\best_model.pth",
    input_dir=r"Z:\bmahsa\workplaces\CellFundusSegmentation\raw data\Full Annotations--New ROIs\Ateam",
    output_dir=r"Z:\bmahsa\workplaces\CellFundusSegmentation\imagej_predicted_rois_1436 0LN stack",
    device="cpu",
    cell_threshold=0.8,
    focus_threshold=0.8,
    min_size=25
)
