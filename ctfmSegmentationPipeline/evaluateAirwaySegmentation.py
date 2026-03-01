import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate airway segmentation with centerline and overlap metrics."
    )
    parser.add_argument("--prediction-path", type=Path, default=None, help="Single-case prediction NIfTI path.")
    parser.add_argument("--ground-truth-path", type=Path, default=None, help="Single-case ground-truth NIfTI path.")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help="Batch mode root containing <caseId>/airwayMask.nii.gz.",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=Path("data/AeroPath"),
        help="AeroPath root with <caseId>/<caseId>_CT_HR_label_airways.nii.gz.",
    )
    parser.add_argument(
        "--prediction-file-name",
        type=str,
        default="airwayMask.nii.gz",
        help="Prediction file name in each case folder for batch mode.",
    )
    parser.add_argument(
        "--exclude-mask-root",
        type=Path,
        default=None,
        help="Optional batch root with <caseId>/<exclude-mask-file-name> to remove from both masks.",
    )
    parser.add_argument(
        "--exclude-mask-file-name",
        type=str,
        default="excludeMask.nii.gz",
        help="Exclude mask file name used with --exclude-mask-root.",
    )
    parser.add_argument(
        "--exclude-mask-path",
        type=Path,
        default=None,
        help="Optional NIfTI mask to remove from both prediction and GT (e.g., trachea+main bronchi).",
    )
    parser.add_argument(
        "--remove-top-slices",
        type=int,
        default=0,
        help="Optional number of axial slices to remove from both masks before scoring.",
    )
    parser.add_argument(
        "--top-side",
        choices=("high", "low"),
        default="high",
        help="If removing top slices, remove from high-z or low-z end.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting prediction to binary if not already binary.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output file.")
    return parser.parse_args()


def main() -> None:
    args = parseArgs()

    if args.prediction_path is not None or args.ground_truth_path is not None:
        if args.prediction_path is None or args.ground_truth_path is None:
            raise ValueError("Single-case mode requires both --prediction-path and --ground-truth-path.")
        result = evaluateCase(
            predictionPath=args.prediction_path,
            groundTruthPath=args.ground_truth_path,
            threshold=args.threshold,
            excludeMaskPath=args.exclude_mask_path,
            removeTopSlices=args.remove_top_slices,
            topSide=args.top_side,
        )
        payload = {"case": result}
    else:
        if args.prediction_root is None:
            raise ValueError("Batch mode requires --prediction-root.")
        perCase = evaluateBatch(
            predictionRoot=args.prediction_root,
            groundTruthRoot=args.ground_truth_root,
            predictionFileName=args.prediction_file_name,
            threshold=args.threshold,
            excludeMaskRoot=args.exclude_mask_root,
            excludeMaskFileName=args.exclude_mask_file_name,
            removeTopSlices=args.remove_top_slices,
            topSide=args.top_side,
        )
        payload = {
            "perCase": perCase,
            "summary": summarizeMetrics(perCase),
        }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Saved metrics to: {args.output_json}")


def evaluateBatch(
    predictionRoot: Path,
    groundTruthRoot: Path,
    predictionFileName: str,
    threshold: float,
    excludeMaskRoot: Path | None,
    excludeMaskFileName: str,
    removeTopSlices: int,
    topSide: str,
) -> list[dict]:
    results: list[dict] = []
    for caseDir in sorted([path for path in predictionRoot.iterdir() if path.is_dir()], key=_caseSortKey):
        caseId = caseDir.name
        predictionPath = caseDir / predictionFileName
        groundTruthPath = groundTruthRoot / caseId / f"{caseId}_CT_HR_label_airways.nii.gz"
        if not predictionPath.exists() or not groundTruthPath.exists():
            continue
        excludeMaskPath = None
        if excludeMaskRoot is not None:
            candidate = excludeMaskRoot / caseId / excludeMaskFileName
            if candidate.exists():
                excludeMaskPath = candidate

        caseResult = evaluateCase(
            predictionPath=predictionPath,
            groundTruthPath=groundTruthPath,
            threshold=threshold,
            excludeMaskPath=excludeMaskPath,
            removeTopSlices=removeTopSlices,
            topSide=topSide,
        )
        caseResult["caseId"] = caseId
        results.append(caseResult)
    return results


def evaluateCase(
    predictionPath: Path,
    groundTruthPath: Path,
    threshold: float,
    excludeMaskPath: Path | None,
    removeTopSlices: int,
    topSide: str,
) -> dict:
    predictionNifti = nib.load(str(predictionPath))
    groundTruthNifti = nib.load(str(groundTruthPath))

    prediction = predictionNifti.get_fdata()
    groundTruth = groundTruthNifti.get_fdata()
    if prediction.shape != groundTruth.shape:
        raise ValueError(
            f"Shape mismatch between prediction {prediction.shape} and GT {groundTruth.shape} for {predictionPath}."
        )

    predictionMask = toBinary(prediction, threshold=threshold)
    groundTruthMask = groundTruth > 0.5

    exclusionMask = np.zeros_like(groundTruthMask, dtype=bool)
    if excludeMaskPath is not None:
        exclusionMask = nib.load(str(excludeMaskPath)).get_fdata() > 0.5
        if exclusionMask.shape != groundTruthMask.shape:
            raise ValueError(
                f"Exclude mask shape {exclusionMask.shape} does not match GT shape {groundTruthMask.shape}."
            )

    if removeTopSlices > 0:
        exclusionMask = exclusionMask | topSliceMask(
            shape=groundTruthMask.shape,
            removeTopSlices=removeTopSlices,
            topSide=topSide,
        )

    predictionMask = predictionMask & ~exclusionMask
    groundTruthMask = groundTruthMask & ~exclusionMask

    spacing = tuple(float(value) for value in groundTruthNifti.header.get_zooms()[:3])
    metrics = computeAirwayMetrics(
        prediction=predictionMask,
        groundTruth=groundTruthMask,
        spacing=spacing,
    )
    metrics["predictionPath"] = str(predictionPath)
    metrics["groundTruthPath"] = str(groundTruthPath)
    return metrics


def computeAirwayMetrics(prediction: np.ndarray, groundTruth: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    prediction = prediction.astype(bool)
    groundTruth = groundTruth.astype(bool)

    predictionCenterline = skeletonize(prediction, method="lee").astype(bool)
    groundTruthCenterline = skeletonize(groundTruth, method="lee").astype(bool)

    gtCenterlineLengthVox = int(groundTruthCenterline.sum())
    if gtCenterlineLengthVox == 0:
        raise ValueError("Ground truth centerline is empty; cannot compute TL/CL.")

    gtTreeInsidePredictionVox = int((groundTruthCenterline & prediction).sum())
    predictionCenterlineOutsideGtVox = int((predictionCenterline & ~groundTruth).sum())

    falsePositiveVox = int((prediction & ~groundTruth).sum())
    groundTruthVox = int(groundTruth.sum())
    intersectionVox = int((prediction & groundTruth).sum())
    predictionVox = int(prediction.sum())

    tlPercent = 100.0 * gtTreeInsidePredictionVox / gtCenterlineLengthVox
    clPercent = 100.0 * predictionCenterlineOutsideGtVox / gtCenterlineLengthVox
    fprPercent = 100.0 * falsePositiveVox / max(groundTruthVox, 1)

    diceDenominator = predictionVox + groundTruthVox
    dice = (2.0 * intersectionVox / diceDenominator) if diceDenominator > 0 else 1.0

    voxelScale = float(np.cbrt(np.prod(np.abs(np.array(spacing, dtype=np.float64)))))
    totalTreeLength = gtTreeInsidePredictionVox * voxelScale

    return {
        "tlPercent": tlPercent,
        "clPercent": clPercent,
        "fprPercent": fprPercent,
        "dice": dice,
        "totalTreeLength": totalTreeLength,
        "totalTreeLengthUnits": "mm",
        "gtCenterlineLengthVox": gtCenterlineLengthVox,
        "gtTreeInsidePredictionVox": gtTreeInsidePredictionVox,
        "predictionCenterlineOutsideGtVox": predictionCenterlineOutsideGtVox,
        "falsePositiveVox": falsePositiveVox,
        "groundTruthVox": groundTruthVox,
        "spacing": spacing,
    }


def summarizeMetrics(perCase: list[dict]) -> dict:
    if not perCase:
        return {"numCases": 0}

    metricNames = ["tlPercent", "clPercent", "fprPercent", "dice", "totalTreeLength"]
    summary: dict[str, float] = {"numCases": len(perCase)}
    for metricName in metricNames:
        values = np.array([float(case[metricName]) for case in perCase], dtype=np.float64)
        summary[f"{metricName}Mean"] = float(np.mean(values))
        summary[f"{metricName}Std"] = float(np.std(values))
    return summary


def toBinary(volume: np.ndarray, threshold: float) -> np.ndarray:
    uniqueValues = np.unique(volume)
    if uniqueValues.size <= 2 and np.all(np.isin(uniqueValues, [0.0, 1.0])):
        return volume > 0.5
    return volume >= threshold


def topSliceMask(shape: tuple[int, int, int], removeTopSlices: int, topSide: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    removeTopSlices = max(0, int(removeTopSlices))
    if removeTopSlices == 0:
        return mask

    depth = shape[2]
    removeTopSlices = min(removeTopSlices, depth)
    if topSide == "high":
        mask[:, :, depth - removeTopSlices :] = True
    else:
        mask[:, :, :removeTopSlices] = True
    return mask


def _caseSortKey(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


if __name__ == "__main__":
    main()
