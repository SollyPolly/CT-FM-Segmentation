import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose,
    CropForeground,
    EnsureType,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
)
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CT-FM embeddings for all cases and plot PCA.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\danny\Documents\Code\Imperial_Dissertation\data\AeroPath"),
        help="Root folder containing AeroPath case subfolders.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_CT_HR.nii.gz",
        help="Pattern used to find input volumes recursively.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        metavar=("D", "H", "W"),
        help="Patch size used during inference.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Patch overlap fraction in [0.0, 0.9).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision on CUDA.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\danny\Documents\Code\Imperial_Dissertation\outputs\ctfm_pca"),
        help="Directory to save features and the PCA plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the matplotlib window.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_preprocess() -> Compose:
    return Compose(
        [
            LoadImage(ensure_channel_first=True),
            EnsureType(),
            Orientation(
                axcodes="SPL",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            ScaleIntensityRange(
                a_min=-1024,
                a_max=2048,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForeground(),
        ]
    )


def get_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last_start = length - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_patch(
    volume: torch.Tensor,
    z: int,
    y: int,
    x: int,
    roi_size: tuple[int, int, int],
) -> tuple[torch.Tensor, int]:
    roi_d, roi_h, roi_w = roi_size
    patch = volume[:, z : z + roi_d, y : y + roi_h, x : x + roi_w]

    valid_d = patch.shape[1]
    valid_h = patch.shape[2]
    valid_w = patch.shape[3]
    valid_voxels = valid_d * valid_h * valid_w

    pad_d = roi_d - valid_d
    pad_h = roi_h - valid_h
    pad_w = roi_w - valid_w
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)

    return patch, valid_voxels


def extract_embedding(
    input_path: Path,
    model: torch.nn.Module,
    preprocess: Compose,
    device: torch.device,
    roi_size: tuple[int, int, int],
    overlap: float,
    use_amp: bool,
) -> np.ndarray:
    volume = preprocess(str(input_path))
    volume = torch.as_tensor(volume, dtype=torch.float32)
    if volume.ndim != 4:
        raise ValueError(f"Expected volume shape [C, D, H, W], got {tuple(volume.shape)} for {input_path}.")

    stride = tuple(max(1, int(size * (1.0 - overlap))) for size in roi_size)
    z_starts = get_starts(volume.shape[1], roi_size[0], stride[0])
    y_starts = get_starts(volume.shape[2], roi_size[1], stride[1])
    x_starts = get_starts(volume.shape[3], roi_size[2], stride[2])

    feature_sum = None
    total_weight = 0.0

    with torch.inference_mode():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch, valid_voxels = extract_patch(volume, z, y, x, roi_size)
                    patch = patch.unsqueeze(0).to(device, non_blocking=True)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        features = model(patch)[-1]
                        pooled = F.adaptive_avg_pool3d(features, 1).flatten()

                    pooled = pooled.detach().to("cpu", dtype=torch.float32)
                    weight = float(valid_voxels)
                    if feature_sum is None:
                        feature_sum = pooled * weight
                    else:
                        feature_sum += pooled * weight
                    total_weight += weight

    if feature_sum is None or total_weight == 0.0:
        raise RuntimeError(f"No patches were processed for {input_path}.")

    return (feature_sum / total_weight).numpy()


def case_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.parent.name), path.name)
    except ValueError:
        return (10**9, path.name)


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not 0.0 <= args.overlap < 0.9:
        raise ValueError("--overlap must be in [0.0, 0.9).")

    input_paths = sorted(args.data_root.rglob(args.glob), key=case_sort_key)
    if len(input_paths) < 2:
        raise RuntimeError(f"Need at least 2 scans for PCA, found {len(input_paths)}.")

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    roi_size = tuple(int(x) for x in args.roi_size)

    preprocess = build_preprocess()
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model = model.to(device)
    model.eval()

    case_ids: list[str] = []
    features: list[np.ndarray] = []

    for idx, input_path in enumerate(input_paths, start=1):
        case_id = input_path.parent.name
        print(f"[{idx}/{len(input_paths)}] Extracting case {case_id}: {input_path}")
        embedding = extract_embedding(
            input_path=input_path,
            model=model,
            preprocess=preprocess,
            device=device,
            roi_size=roi_size,
            overlap=args.overlap,
            use_amp=use_amp,
        )
        case_ids.append(case_id)
        features.append(embedding)

    X = np.stack(features, axis=0)
    Z = PCA(n_components=2).fit_transform(X)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.out_dir / "features.npy"
    case_ids_path = args.out_dir / "case_ids.txt"
    plot_path = args.out_dir / "pca.png"

    np.save(features_path, X)
    case_ids_path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")

    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=50)
    for case_id, x, y in zip(case_ids, Z[:, 0], Z[:, 1]):
        plt.annotate(case_id, (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of CT-FM embeddings")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)

    print(f"Saved features to: {features_path}")
    print(f"Saved case ids to: {case_ids_path}")
    print(f"Saved PCA plot to: {plot_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
