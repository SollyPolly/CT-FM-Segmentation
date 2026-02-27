import argparse
from pathlib import Path

import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_slices(folder: Path):
    files = sorted(folder.glob('*.dcm'))
    if not files:
        raise FileNotFoundError(f'No .dcm files found in {folder}')

    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=False)
        instance = getattr(ds, 'InstanceNumber', None)
        slices.append((instance, f, ds))

    # Sort by InstanceNumber when available; otherwise by filename
    if all(s[0] is not None for s in slices):
        slices.sort(key=lambda x: x[0])
    else:
        slices.sort(key=lambda x: x[1].name)

    images = []
    for _, _, ds in slices:
        img = ds.pixel_array.astype(np.float32)
        # Apply RescaleSlope/Intercept if present
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        img = img * slope + intercept
        images.append(img)

    return images


def main():
    parser = argparse.ArgumentParser(description='Simple DICOM slice viewer (scroll/slider).')
    parser.add_argument('folder', type=str, help='Path to a folder containing .dcm slices')
    args = parser.parse_args()

    folder = Path(args.folder)
    images = load_slices(folder)

    idx = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    im = ax.imshow(images[idx], cmap='gray')
    ax.set_title(f'Slice {idx + 1} / {len(images)}')
    ax.axis('off')

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
    slider = Slider(ax_slider, 'Slice', 1, len(images), valinit=1, valstep=1)

    def update(i):
        nonlocal idx
        idx = int(i) - 1
        im.set_data(images[idx])
        ax.set_title(f'Slice {idx + 1} / {len(images)}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_scroll(event):
        nonlocal idx
        if event.button == 'up':
            idx = min(idx + 1, len(images) - 1)
        elif event.button == 'down':
            idx = max(idx - 1, 0)
        slider.set_val(idx + 1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()


if __name__ == '__main__':
    main()
