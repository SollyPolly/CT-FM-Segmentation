import argparse
from pathlib import Path

import numpy as np
import pydicom
import nibabel as nib


def load_dicoms(folder: Path):
    files = sorted(folder.glob('*.dcm'))
    if not files:
        raise FileNotFoundError(f'No .dcm files found in {folder}')

    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        ipp = getattr(ds, 'ImagePositionPatient', None)
        inst = getattr(ds, 'InstanceNumber', None)
        slices.append((ipp, inst, f, ds))

    # Prefer sorting by ImagePositionPatient along slice normal; fallback to InstanceNumber/name
    if all(s[0] is not None for s in slices):
        iop = np.array(getattr(slices[0][3], 'ImageOrientationPatient'))
        row_cos = iop[:3]
        col_cos = iop[3:]
        normal = np.cross(row_cos, col_cos)
        def pos_key(s):
            return float(np.dot(np.array(s[0]), normal))
        slices.sort(key=pos_key)
    elif all(s[1] is not None for s in slices):
        slices.sort(key=lambda x: x[1])
    else:
        slices.sort(key=lambda x: x[2].name)

    return slices


def build_affine(ds0, ds1=None):
    # Pixel spacing in mm
    px_spacing = np.array(getattr(ds0, 'PixelSpacing', [1.0, 1.0]), dtype=float)
    row_spacing, col_spacing = px_spacing

    iop = np.array(getattr(ds0, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0]), dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:]
    normal = np.cross(row_cos, col_cos)

    # Slice spacing: prefer SpacingBetweenSlices or SliceThickness, else compute from position
    slice_spacing = float(getattr(ds0, 'SpacingBetweenSlices', getattr(ds0, 'SliceThickness', 1.0)))
    if ds1 is not None and hasattr(ds0, 'ImagePositionPatient') and hasattr(ds1, 'ImagePositionPatient'):
        p0 = np.array(ds0.ImagePositionPatient, dtype=float)
        p1 = np.array(ds1.ImagePositionPatient, dtype=float)
        slice_spacing = float(abs(np.dot(p1 - p0, normal)))

    # DICOM: ImagePositionPatient is the origin of the first voxel (row=0,col=0)
    origin = np.array(getattr(ds0, 'ImagePositionPatient', [0.0, 0.0, 0.0]), dtype=float)

    # Affine columns are direction vectors scaled by spacing
    affine = np.eye(4, dtype=float)
    affine[:3, 0] = row_cos * row_spacing
    affine[:3, 1] = col_cos * col_spacing
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = origin
    return affine


def main():
    parser = argparse.ArgumentParser(description='Convert DICOM slice folder to NIfTI .nii.gz')
    parser.add_argument('folder', type=str, help='Path to a folder containing .dcm slices')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output .nii.gz path')
    args = parser.parse_args()

    folder = Path(args.folder)
    slices = load_dicoms(folder)

    # Build 3D volume
    images = []
    for _, _, _, ds in slices:
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images, axis=-1)

    ds0 = slices[0][3]
    ds1 = slices[1][3] if len(slices) > 1 else None
    affine = build_affine(ds0, ds1)

    nii = nib.Nifti1Image(volume, affine)

    if args.output:
        output = Path(args.output)
    else:
        out_dir = Path('data') / 'osic_nifti'
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f'{folder.name}.nii.gz'
    nib.save(nii, str(output))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
