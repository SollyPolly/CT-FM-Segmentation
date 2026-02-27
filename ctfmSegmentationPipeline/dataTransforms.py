from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)

from ctfmSegmentationPipeline.config import SegmentationConfig


def buildTrainTransforms(config: SegmentationConfig) -> Compose:
    includeLungMask = config.useLungRoi
    keys = _availableKeys(config=config, includeLabel=True, includeLungMask=includeLungMask)
    cropKeys = [config.imageKey, config.labelKey]
    augmentKeys = [config.imageKey, config.labelKey]
    if includeLungMask:
        cropKeys.append(config.lungMaskKey)
        augmentKeys.append(config.lungMaskKey)
    transforms = _buildCommonTransforms(config=config, keys=keys)
    transforms.extend(
        [
            RandCropByPosNegLabeld(
                keys=cropKeys,
                label_key=config.labelKey,
                spatial_size=config.patchSize,
                pos=1.0,
                neg=1.0,
                num_samples=config.samplesPerCase,
                image_key=config.imageKey,
                image_threshold=0.0,
                allow_smaller=True,
            ),
            RandFlipd(keys=augmentKeys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=augmentKeys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=augmentKeys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=augmentKeys, prob=0.3, max_k=3),
        ]
    )
    return Compose(transforms)


def buildValTransforms(config: SegmentationConfig) -> Compose:
    keys = _availableKeys(config=config, includeLabel=True, includeLungMask=config.useLungRoi)
    return Compose(_buildCommonTransforms(config=config, keys=keys))


def buildInferenceTransforms(config: SegmentationConfig, includeLungMask: bool) -> Compose:
    keys = [config.imageKey]
    if includeLungMask:
        keys.append(config.lungMaskKey)
    return Compose(_buildCommonTransforms(config=config, keys=keys))


def _buildCommonTransforms(config: SegmentationConfig, keys: list[str]) -> list:
    transforms: list = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
        EnsureTyped(keys=keys),
    ]

    if config.normalizeOrientation:
        transforms.append(
            Orientationd(
                keys=keys,
                axcodes="SPL",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            )
        )

    if config.useSpacing:
        mode = ["bilinear" if key == config.imageKey else "nearest" for key in keys]
        transforms.append(Spacingd(keys=keys, pixdim=config.spacing, mode=mode))

    transforms.append(
        ScaleIntensityRanged(
            keys=config.imageKey,
            a_min=-1024,
            a_max=2048,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    if config.useLungRoi and config.lungMaskKey in keys:
        transforms.append(CropForegroundd(keys=keys, source_key=config.lungMaskKey))

    return transforms


def _availableKeys(config: SegmentationConfig, includeLabel: bool, includeLungMask: bool) -> list[str]:
    keys = [config.imageKey]
    if includeLabel:
        keys.append(config.labelKey)
    if includeLungMask:
        keys.append(config.lungMaskKey)
    return keys
