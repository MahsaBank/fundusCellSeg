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

## Output

For each microscopy stack, the pipeline produces:

* `*_cell_prob.npy`
* `*_focus_prob.npy`
* `*_binary.npy`
* `*_labels.npy`
* `*_predicted_cells.zip`

The ROI ZIP file can be opened directly in ImageJ/Fiji for visualization and downstream quantitative analysis.
