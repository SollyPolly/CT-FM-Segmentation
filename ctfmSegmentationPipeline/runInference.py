import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from ctfmSegmentationPipeline.aeroPathDataset import buildInferenceDataset
from ctfmSegmentationPipeline.config import SegmentationConfig
from ctfmSegmentationPipeline.pathUtils import ensureExperimentDirs
from ctfmSegmentationPipeline.segmentationModel import buildSegmentationModel


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run airway segmentation inference from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--lung-mask-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ctfmSegmentation/inference"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = SegmentationConfig.fromDict(checkpoint["config"])

    if args.device is not None:
        config.device = args.device
    if args.threshold is not None:
        config.threshold = args.threshold

    if config.normalizeOrientation or config.useSpacing or config.useLungRoi:
        raise RuntimeError(
            "This baseline inference script saves outputs in the input image space. "
            "Disable orientation normalization, spacing resampling, and lung ROI cropping "
            "for checkpoint export, or add an inverse-transform save path first."
        )

    model, _ = buildSegmentationModel(config)
    model.load_state_dict(checkpoint["modelState"])
    model = model.to(config.device)
    model.eval()

    record = {
        "caseId": args.input_path.stem.replace(".nii", ""),
        config.imageKey: str(args.input_path),
        config.lungMaskKey: str(args.lung_mask_path) if args.lung_mask_path is not None else None,
    }
    dataset = buildInferenceDataset(config=config, record=record)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.inference_mode():
        batch = next(iter(loader))
        imageTensor = batch[config.imageKey].to(config.device)
        logits = sliding_window_inference(
            inputs=imageTensor,
            roi_size=config.patchSize,
            sw_batch_size=config.inferenceBatchSize,
            predictor=model,
            overlap=config.inferenceOverlap,
        )
        probabilityVolume = torch.sigmoid(logits)[0, 0].cpu().numpy()
        binaryVolume = (probabilityVolume >= config.threshold).astype(np.uint8)

    outputDir = args.output_dir / args.input_path.parent.name
    ensureExperimentDirs(outputDir.parent)
    outputDir.mkdir(parents=True, exist_ok=True)

    inputNifti = nib.load(str(args.input_path))
    probabilityPath = outputDir / "airwayProbability.nii.gz"
    binaryPath = outputDir / "airwayMask.nii.gz"
    npyPath = outputDir / "airwayProbability.npy"

    nib.save(nib.Nifti1Image(probabilityVolume.astype(np.float32), inputNifti.affine), str(probabilityPath))
    nib.save(nib.Nifti1Image(binaryVolume, inputNifti.affine), str(binaryPath))
    np.save(npyPath, probabilityVolume)

    print(f"Saved probability map to: {probabilityPath}")
    print(f"Saved binary mask to: {binaryPath}")
    print(f"Saved raw probabilities to: {npyPath}")


if __name__ == "__main__":
    main()
