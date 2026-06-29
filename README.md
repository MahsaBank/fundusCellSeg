# Cell Segmentation and ImageJ ROI Generation Pipeline

## Overview

This repository provides an end-to-end deep learning pipeline for automated cell detection in fluorescence microscopy z-stacks. The framework performs Attention U-Net inference, converts semantic predictions into individual cell instances, associates corresponding cells across neighboring z-slices, and exports the final detections as ImageJ-compatible ROI files.

The pipeline is designed to integrate directly with ImageJ/Fiji, allowing researchers to review, edit, and quantify automatically detected cells using familiar annotation workflows.

---

## Repository Structure

```text
.
├── train_model_v5.py
├── cell_segmentation_roi_pipeline.py
├── inference_v5.py
└── README.md
```

---

## Scripts

### train_model_v5.py

Trains the Attention U-Net model using manually annotated fluorescence microscopy images.

Main tasks:

* Data loading and preprocessing
* Intensity normalization
* Model training and validation
* Checkpoint generation

---

### cell_segmentation_roi_pipeline.py (Main Pipeline)

This is the primary script in the repository. Starting from raw fluorescence microscopy TIFF stacks, it performs the complete workflow:

1. Loads a trained Attention U-Net model.
2. Performs inference to generate cell and focus probability maps.
3. Converts probability maps into instance segmentations using thresholding and watershed segmentation.
4. Associates cells across neighboring z-slices using histogram similarity.
5. Generates ImageJ-compatible oval ROIs.
6. Saves probability maps, binary masks, instance labels, and ImageJ ROI ZIP files.

For most users, this is the only script needed to analyze new microscopy datasets.

---

### inference_v5.py

A simplified inference script that runs the trained network and produces probability maps without performing ROI generation or postprocessing.

This script is useful for:

* Inspecting network predictions
* Evaluating probability maps
* Developing alternative postprocessing methods

---

## Pipeline

```text
Input TIFF Stack
        │
        ▼
Attention U-Net Inference
        │
        ▼
Cell & Focus Probability Maps
        │
        ▼
Thresholding
        │
        ▼
Small Object Removal
        │
        ▼
Distance Transform
        │
        ▼
Peak Local Max
        │
        ▼
Watershed Instance Segmentation
        │
        ▼
Cross-Slice Cell Association
        │
        ▼
ImageJ ROI Generation
        │
        ▼
ROI ZIP + Probability Maps + Labels
```

---

## Typical Usage

### Train a model

```bash
python train_model_v5.py
```

### Run the complete prediction pipeline

```bash
python cell_segmentation_roi_pipeline.py
```

This command generates:

* Cell probability maps
* Focus probability maps
* Binary segmentation masks
* Instance label maps
* ImageJ ROI ZIP files

---

## Main Pipeline Parameters

| Parameter         | Description                                                                                                                                                                         |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_dir`       | Directory containing input fluorescence microscopy TIFF stacks (`*.tif`).                                                                                                           |
| `output_dir`      | Directory where prediction results, probability maps, and ImageJ ROI ZIP files will be saved.                                                                                       |
| `chk_filename`    | Path to the trained Attention U-Net model checkpoint (`.pth`).                                                                                                                      |
| `device`          | Computation device used for inference (`"cpu"` or `"cuda"`).                                                                                                                        |
| `cell_threshold`  | Probability threshold applied to the predicted cell probability map before instance segmentation. Higher values produce more conservative detections.                               |
| `focus_threshold` | Probability threshold applied to the predicted focus map. The focus predictions are generated and saved, and can optionally be used to suppress detections outside focused regions. |
| `min_size`        | Minimum connected-component size (in pixels) retained after thresholding. Smaller components are removed as noise before watershed segmentation.                                    |

### Example

```python
run_prediction_to_imagej_rois(
    chk_filename="best_model.pth",
    input_dir="example_data/",
    output_dir="prediction_results/",
    device="cuda",
    cell_threshold=0.8,
    focus_threshold=0.8,
    min_size=25,
)
```

### Generated Outputs

For each input image stack, the pipeline generates:

* `*_predicted_cells.zip` — ImageJ-compatible oval ROIs.
* `*_cell_prob.npy` — Predicted cell probability map.
* `*_focus_prob.npy` — Predicted focus probability map.
* `*_binary.npy` — Binary segmentation mask after thresholding and postprocessing.
* `*_labels.npy` — Watershed-generated instance labels.
  
The ROI ZIP file can be opened directly in ImageJ/Fiji for visualization and downstream quantitative analysis.

Note: The current implementation assumes that each TIFF stack has the shape (Z, P, H, W), where Z is the number of z-slices and P=2 corresponds to two fluorescence frames (or channels) used as the network input. This documents an important assumption that's currently implicit in your code.
