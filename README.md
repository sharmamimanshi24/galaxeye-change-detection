# Binary Change Detection on EO–SAR Image Pairs

GalaxEye Satellite AI Research Intern — Technical Assignment
Author: Mimanshi Sharma

## Project Description

This repository implements a binary pixel-level change detection model for co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) image pairs from a building-damage assessment dataset. The pre-event image is 3-band optical (RGB) and the post-event image is single-band SAR, making this a cross-modal change detection task.

The model is a U-Net with a ResNet-34 encoder pretrained on ImageNet, adapted to take 4 input channels (3 EO + 1 SAR concatenated via early fusion). Training uses a Dice + binary cross-entropy loss to handle the extreme class imbalance (1.57% change pixels in the training set).

**Headline results:**

| Split | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | 0.4483 | 0.6095 | 0.6289 | 0.6190 |
| Test (visible 50%) | 0.0585 | 0.1145 | 0.1069 | 0.1106 |

The val–test gap is structural: train and val share scenes 01–08, while the visible test set contains only scenes 09 and 10. See the technical report for full analysis.

## Requirements

- Python 3.12
- CUDA-capable GPU (training was run on a Quadro RTX 8000, 48 GB VRAM)
- See `requirements.txt` for pinned dependencies

## Environment Setup

```bash
# Clone the repo
git clone https://github.com/sharmamimanshi24/galaxeye-change-detection.git
cd galaxeye-change-detection

# Create a virtual environment (recommended: on a disk with plenty of free space)
python3 -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` doesn't see CUDA after the default install, reinstall it with the CUDA-specific index URL:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Dataset Structure

Download the dataset from the assignment Data Link and extract such that the layout is:

```
<DATA_ROOT>/
├── train/
│   ├── pre-event/      # 3-band EO TIFFs, 1024×1024 uint8
│   ├── post-event/     # 1-band SAR TIFFs, 1024×1024 uint8
│   └── target/         # 1-band mask TIFFs with values {0,1,2,3}
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Each pre/post/target triplet shares an identical filename. The mask label remap (0,1 → 0; 2,3 → 1) is applied inside the data loader.

| Split | Samples | Scenes |
|---|---|---|
| Train | 2,781 | 01–08 |
| Val | 334 | 01–08 |
| Test | 77 | 09–10 |

## Training

Update the `DATA_ROOT` and `CHECKPOINT_DIR` paths in `config.yaml` to match your environment, then run the training notebook end-to-end:

```bash
jupyter lab
# Open galaxeye_change_detection.ipynb and run all cells
```

Approximate training time: ~15 minutes for 20 epochs on a Quadro RTX 8000.

## Evaluation

The notebook includes an evaluation block that loads the best checkpoint and computes IoU / Precision / Recall / F1 plus the confusion matrix on the test split, and writes a 5-sample qualitative figure (worst → best per-sample IoU) to `qualitative_predictions.png`.

To evaluate on arbitrary data: edit `DATA_ROOT` to point at the test directory, set `CHECKPOINT_DIR` to the location of `best_model.pth`, and re-run the evaluation cells.

## Model Weights

The best V1 checkpoint is hosted publicly on Google Drive:

**[Download best_model.pth](https://drive.google.com/file/d/16nBZgwbsl4Hxzo_tqXXzyZMjd9WGQ-RV/view?usp=sharing)**

Place it under your `CHECKPOINT_DIR` before running evaluation.

## Results

See the technical report PDF in the submission ZIP for the full results, confusion matrix, per-sample distribution, qualitative analysis, and discussion of the validation–test gap.

## References

Key papers consulted (full list in the technical report):

- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. ICIP 2018.
- Zhang, C. et al. (2020). A deeply supervised image fusion network for change detection. ISPRS J. Photogramm. Remote Sens.
- Fang, S. et al. (2021). SNUNet-CD: A densely connected siamese network for change detection of VHR images. IEEE GRSL.
- Chen, H., Qi, Z., & Shi, Z. (2021). Remote sensing image change detection with transformers. IEEE TGRS.
- Bandara, W. G. C., & Patel, V. M. (2022). A transformer-based siamese network for change detection. IGARSS 2022.
- Liu, Z. et al. (2025). M²CD: A unified multimodal framework for optical-SAR change detection. arXiv:2503.19406.
- Chen, H. et al. (2020). Deep siamese domain adaptation CNN for cross-domain change detection. arXiv:2004.05745.
- Gupta, R. et al. (2020). RescueNet: Joint building segmentation and damage assessment. arXiv:2004.07312.
- Shen, Y. et al. (2021). BDANet: Multiscale CNN with cross-directional attention for building damage. arXiv:2105.07364.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI 2015.
- He, K. et al. (2016). Deep residual learning for image recognition. CVPR 2016.
- Iakubovskii, P. (2019). segmentation_models_pytorch. https://github.com/qubvel/segmentation_models.pytorch
