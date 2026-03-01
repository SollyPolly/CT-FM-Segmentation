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
- Training checkpoints are saved on best validation Dice (`bestModel.pt`) and latest epoch (`lastModel.pt`).
- Validation logs now include:
  - `valDice`
  - `valPrecision`
  - `valRecall`
  - `valIoU`
  - `valHd95Vox`
  - `valAvgSurfaceDistVox`

## Cross-Validation + Holdout

Reserve a holdout subset and run k-fold cross-validation on the remaining development split:

```powershell
python -m ctfmSegmentationPipeline.crossValidateModel `
  --data-root .\data\AeroPath `
  --experiment-name baselineSegResNetCv `
  --device cuda --amp `
  --num-folds 5 `
  --holdout-fraction 0.15 `
  --max-epochs 20 `
  --train-batch-size 1 `
  --patch-size 96 96 96 `
  --num-workers 2
```

Summary files are written under:
- `outputs/ctfmSegmentation/<experiment>_crossVal/splitManifest.json`
- `outputs/ctfmSegmentation/<experiment>_crossVal/crossValidationSummary.json`

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

## Airway-Specific Evaluation

The repository now includes centerline-aware airway metrics:
- `tlPercent`: tree length captured, i.e. GT centerline voxels inside prediction / full GT centerline length.
- `clPercent`: centerline leakage, i.e. predicted centerline voxels outside GT / full GT centerline length.
- `fprPercent`: false-positive voxels outside GT / total GT voxels.
- `dice`: voxel overlap Dice.
- `totalTreeLength`: captured GT centerline length scaled by geometric-mean voxel size.

Single-case example:

```powershell
python -m ctfmSegmentationPipeline.evaluateAirwaySegmentation `
  --prediction-path .\outputs\ctfmSegmentation\inference\1\airwayMask.nii.gz `
  --ground-truth-path .\data\AeroPath\1\1_CT_HR_label_airways.nii.gz `
  --output-json .\outputs\ctfmSegmentation\inference\1\metrics.json
```

Batch example (all case folders under inference root):

```powershell
python -m ctfmSegmentationPipeline.evaluateAirwaySegmentation `
  --prediction-root .\outputs\ctfmSegmentation\inference `
  --ground-truth-root .\data\AeroPath `
  --output-json .\outputs\ctfmSegmentation\inference\metrics_summary.json
```

If you want to remove trachea/main bronchi from both prediction and GT before scoring:
- single case: use `--exclude-mask-path <path>`
- batch: use `--exclude-mask-root <root>` and per-case `<caseId>/<exclude-mask-file-name>`

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
