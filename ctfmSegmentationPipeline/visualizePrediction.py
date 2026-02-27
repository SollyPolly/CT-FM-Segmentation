import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a CT volume with a segmentation overlay.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--axis", choices=("axial", "coronal", "sagittal"), default="axial")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    imageVolume = nib.load(str(args.image_path)).get_fdata()
    maskVolume = nib.load(str(args.mask_path)).get_fdata()

    if imageVolume.shape != maskVolume.shape:
        raise ValueError(f"Image and mask shapes must match, got {imageVolume.shape} and {maskVolume.shape}.")

    sliceIndex = args.slice_index
    if sliceIndex is None:
        axisLength = _axisLength(imageVolume, args.axis)
        sliceIndex = axisLength // 2

    imageSlice = _extractSlice(imageVolume, args.axis, sliceIndex)
    maskSlice = _extractSlice(maskVolume, args.axis, sliceIndex)

    plt.figure(figsize=(8, 8))
    plt.imshow(imageSlice, cmap="gray")
    plt.imshow(np.ma.masked_where(maskSlice <= 0.0, maskSlice), cmap="autumn", alpha=args.alpha)
    plt.title(f"{args.axis} slice {sliceIndex}")
    plt.axis("off")

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {args.output_path}")
    else:
        plt.show()


def _extractSlice(volume: np.ndarray, axis: str, sliceIndex: int) -> np.ndarray:
    if axis == "axial":
        return np.rot90(volume[:, :, sliceIndex])
    if axis == "coronal":
        return np.rot90(volume[:, sliceIndex, :])
    return np.rot90(volume[sliceIndex, :, :])


def _axisLength(volume: np.ndarray, axis: str) -> int:
    if axis == "axial":
        return volume.shape[2]
    if axis == "coronal":
        return volume.shape[1]
    return volume.shape[0]


if __name__ == "__main__":
    main()
