import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot airway probability map over CT with optional lung contour."
    )
    parser.add_argument("--image-path", type=Path, required=True, help="Path to input CT NIfTI.")
    parser.add_argument(
        "--probability-path",
        type=Path,
        required=True,
        help="Path to airway probability NIfTI.",
    )
    parser.add_argument(
        "--lung-mask-path",
        type=Path,
        default=None,
        help="Optional lung mask NIfTI for contour overlay.",
    )
    parser.add_argument(
        "--axis",
        choices=("axial", "coronal", "sagittal"),
        default="axial",
        help="Slice orientation for plotting.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Explicit slice index in the selected axis. If omitted, slice is auto-selected.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=("max_prob", "center"),
        default="max_prob",
        help="Automatic slice selection strategy when --slice-index is omitted.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for probability heatmap.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap for probability map.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output PNG path. If omitted, opens interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_volume = nib.load(str(args.image_path)).get_fdata()
    prob_volume = nib.load(str(args.probability_path)).get_fdata()

    if image_volume.shape != prob_volume.shape:
        raise ValueError(
            f"Image and probability shapes must match, got {image_volume.shape} and {prob_volume.shape}."
        )

    lung_volume = None
    if args.lung_mask_path is not None:
        lung_volume = nib.load(str(args.lung_mask_path)).get_fdata()
        if lung_volume.shape != image_volume.shape:
            raise ValueError(
                f"Lung mask shape must match image shape, got {lung_volume.shape} and {image_volume.shape}."
            )

    slice_index = args.slice_index
    if slice_index is None:
        slice_index = auto_slice_index(prob_volume=prob_volume, axis=args.axis, mode=args.slice_mode)

    axis_length = axis_size(image_volume, args.axis)
    if slice_index < 0 or slice_index >= axis_length:
        raise ValueError(
            f"Slice index {slice_index} out of bounds for axis '{args.axis}' with size {axis_length}."
        )

    ct_slice = extract_slice(image_volume, args.axis, slice_index)
    prob_slice = extract_slice(prob_volume, args.axis, slice_index)
    lung_slice = extract_slice(lung_volume, args.axis, slice_index) if lung_volume is not None else None

    v_min, v_max = np.percentile(ct_slice, [1, 99])

    plt.figure(figsize=(8, 8))
    plt.imshow(ct_slice, cmap="gray", vmin=v_min, vmax=v_max)
    overlay = plt.imshow(prob_slice, cmap=args.cmap, vmin=0.0, vmax=1.0, alpha=args.alpha)
    plt.colorbar(overlay, fraction=0.046, pad=0.04, label="Airway probability")

    if lung_slice is not None:
        plt.contour(lung_slice, levels=[0.5], colors="cyan", linewidths=0.8)

    plt.title(f"{args.axis.capitalize()} slice {slice_index}")
    plt.axis("off")
    plt.tight_layout()

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output_path, dpi=220, bbox_inches="tight")
        print(f"Saved figure to: {args.output_path}")
    else:
        plt.show()


def extract_slice(volume: np.ndarray | None, axis: str, slice_index: int) -> np.ndarray | None:
    if volume is None:
        return None
    if axis == "axial":
        return np.rot90(volume[:, :, slice_index])
    if axis == "coronal":
        return np.rot90(volume[:, slice_index, :])
    return np.rot90(volume[slice_index, :, :])


def axis_size(volume: np.ndarray, axis: str) -> int:
    if axis == "axial":
        return volume.shape[2]
    if axis == "coronal":
        return volume.shape[1]
    return volume.shape[0]


def auto_slice_index(prob_volume: np.ndarray, axis: str, mode: str) -> int:
    if mode == "center":
        return axis_size(prob_volume, axis) // 2

    if axis == "axial":
        profile = prob_volume.sum(axis=(0, 1))
    elif axis == "coronal":
        profile = prob_volume.sum(axis=(0, 2))
    else:
        profile = prob_volume.sum(axis=(1, 2))

    if float(profile.max()) <= 0.0:
        return axis_size(prob_volume, axis) // 2
    return int(np.argmax(profile))


if __name__ == "__main__":
    main()
