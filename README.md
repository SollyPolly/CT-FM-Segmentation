# CT-FM Segmentation Pipeline (Imperial Dissertation)

This repository contains a supervised airway segmentation baseline for AeroPath using a MONAI `SegResNet` whose encoder is initialized from CT-FM weights.

## Current Pipeline

- Model: `SegResNet` (3D)
- Encoder initialization: CT-FM weights via `lighter_zoo`
- Task: airway segmentation (`*_CT_HR_label_airways.nii.gz`)
- Dataset layout: `data/AeroPath/<caseId>/...`

Main package:
- `ctfmSegmentationPipeline/`

Utility scripts:
- `scripts/featureExtractorCT-FM.py`
- `scripts/featureExtractorCT-FM_optimzied.py`
- `scripts/plot_ctfm_pca.py`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m ctfmSegmentationPipeline.trainModel --data-root .\data\AeroPath --device cuda --amp
```

Notes:
- Default `maxEpochs` is `30`.
- Validation uses full-volume sliding-window inference and can be slow.
- Training checkpoints are saved on best validation Dice.

## Training Outputs

Expected outputs under:
- `outputs/ctfmSegmentation/baselineSegResNet/`

Files:
- `bestModel.pt`
- `history.json`
- `config.json`

## Inference

```powershell
python -m ctfmSegmentationPipeline.runInference `
  --checkpoint .\outputs\ctfmSegmentation\baselineSegResNet\bestModel.pt `
  --input-path .\data\AeroPath\1\1_CT_HR.nii.gz
```

Inference outputs are written under:
- `outputs/ctfmSegmentation/inference/<caseId>/`

## Visualize Prediction

```powershell
python -m ctfmSegmentationPipeline.visualizePrediction `
  --image-path .\data\AeroPath\1\1_CT_HR.nii.gz `
  --mask-path .\outputs\ctfmSegmentation\inference\1\airwayMask.nii.gz
```

## CT-FM Feature Extraction Utilities

Single-volume embedding:

```powershell
python .\scripts\featureExtractorCT-FM_optimzied.py `
  --input-path .\data\AeroPath\1\1_CT_HR.nii.gz `
  --device cuda --amp --plot
```

Dataset-wide PCA:

```powershell
python .\scripts\plot_ctfm_pca.py --data-root .\data\AeroPath --device cuda --amp
```

## Repository Hygiene

Large assets are intentionally ignored by `.gitignore`, including:
- `data/`
- `outputs/`
- `Papers/`
- `CT-FM/`

This keeps the code repository lightweight for GitHub and HPC workflows.
