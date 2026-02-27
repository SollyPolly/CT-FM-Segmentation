import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-optimized CT-FM feature extraction.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(r"C:\Users\danny\Documents\Code\Imperial_Dissertation\data\AeroPath\1\1_CT_HR.nii.gz"),
        help="Path to the input 3D NIfTI volume.",
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
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the final feature vector as a .pt or .npy file.",
    )
    parser.add_argument(
        "--features-npy",
        type=Path,
        default=None,
        help="Optional path to a [n_scans, 512] NumPy array for PCA/similarity plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a histogram of the extracted feature vector.",
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


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input_path}")

    if not 0.0 <= args.overlap < 0.9:
        raise ValueError("--overlap must be in [0.0, 0.9).")

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    preprocess = build_preprocess()
    volume = preprocess(str(args.input_path))
    volume = torch.as_tensor(volume, dtype=torch.float32)

    if volume.ndim != 4:
        raise ValueError(f"Expected volume shape [C, D, H, W], got {tuple(volume.shape)}.")

    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model = model.to(device)
    model.eval()

    roi_size = tuple(int(x) for x in args.roi_size)
    stride = tuple(max(1, int(size * (1.0 - args.overlap))) for size in roi_size)

    z_starts = get_starts(volume.shape[1], roi_size[0], stride[0])
    y_starts = get_starts(volume.shape[2], roi_size[1], stride[1])
    x_starts = get_starts(volume.shape[3], roi_size[2], stride[2])

    feature_sum = None
    total_weight = 0.0
    patch_count = 0

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
                    patch_count += 1

    if feature_sum is None or total_weight == 0.0:
        raise RuntimeError("No patches were processed.")

    avg_output = feature_sum / total_weight

    print("Feature extraction completed")
    print(f"Input: {args.input_path}")
    print(f"Device: {device}")
    print(f"Volume shape after preprocessing: {tuple(volume.shape)}")
    print(f"ROI size: {roi_size}")
    print(f"Patches processed: {patch_count}")
    print(f"Output shape: {tuple(avg_output.shape)}")

    if args.save_path is not None:
        import numpy as np

        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        if args.save_path.suffix.lower() == ".npy":
            np.save(args.save_path, avg_output.numpy())
        else:
            torch.save(avg_output, args.save_path)
        print(f"Saved features to: {args.save_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.hist(avg_output.numpy(), bins=100)
        plt.title("CT-FM feature distribution")
        plt.xlabel("Feature value")
        plt.ylabel("Count")

        if args.features_npy is not None:
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.metrics.pairwise import cosine_similarity

            if not args.features_npy.exists():
                raise FileNotFoundError(f"features array not found: {args.features_npy}")

            X = np.load(args.features_npy)
            if X.ndim != 2 or X.shape[1] != avg_output.numel():
                raise ValueError(
                    f"Expected features array shape [n_scans, {avg_output.numel()}], got {X.shape}."
                )

            Z = PCA(n_components=2).fit_transform(X)
            plt.figure()
            plt.scatter(Z[:, 0], Z[:, 1])
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA of CT-FM embeddings")

            S = cosine_similarity(X)
            plt.figure()
            plt.imshow(S, cmap="viridis")
            plt.title("Cosine similarity")
            plt.colorbar()

        plt.show()

if __name__ == "__main__":
    main()

