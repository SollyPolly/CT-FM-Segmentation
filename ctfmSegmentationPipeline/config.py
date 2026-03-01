from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class SegmentationConfig:
    dataRoot: Path = Path("data/AeroPath")
    outputRoot: Path = Path("outputs/ctfmSegmentation")
    experimentName: str = "baselineSegResNet"

    imageKey: str = "image"
    labelKey: str = "label"
    lungMaskKey: str = "lungMask"

    trainFraction: float = 0.8
    randomSeed: int = 7

    patchSize: tuple[int, int, int] = (96, 96, 96)
    samplesPerCase: int = 2
    trainBatchSize: int = 2
    valBatchSize: int = 1
    numWorkers: int = 0
    cacheRate: float = 0.0

    maxEpochs: int = 30
    validateEvery: int = 1
    freezeEncoderEpochs: int = 5
    encoderLearningRate: float = 1e-4
    decoderLearningRate: float = 3e-4
    weightDecay: float = 1e-5
    gradClipNorm: float | None = 1.0
    earlyStoppingPatience: int = 6
    earlyStoppingMinDelta: float = 0.0
    lrSchedulerFactor: float = 0.5
    lrSchedulerPatience: int = 2
    minLearningRate: float = 1e-6

    inferenceBatchSize: int = 1
    inferenceOverlap: float = 0.5
    threshold: float = 0.5
    computeSurfaceMetrics: bool = True

    normalizeOrientation: bool = False
    useLungRoi: bool = False
    useSpacing: bool = False
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    affineAugmentProbability: float = 0.2
    affineRotateRange: tuple[float, float, float] = (0.1, 0.1, 0.1)
    affineScaleRange: tuple[float, float, float] = (0.1, 0.1, 0.1)
    noiseAugmentProbability: float = 0.15
    intensityScaleAugmentProbability: float = 0.15
    intensityShiftAugmentProbability: float = 0.15

    useCtfmWeights: bool = True
    ctfmModelId: str = "project-lighter/ct_fm_feature_extractor"

    useAmp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def experimentRoot(self) -> Path:
        return self.outputRoot / self.experimentName

    def checkpointPath(self) -> Path:
        return self.experimentRoot() / "bestModel.pt"

    def lastCheckpointPath(self) -> Path:
        return self.experimentRoot() / "lastModel.pt"

    def historyPath(self) -> Path:
        return self.experimentRoot() / "history.json"

    def configPath(self) -> Path:
        return self.experimentRoot() / "config.json"

    def toDict(self) -> dict:
        payload = asdict(self)
        payload["dataRoot"] = str(self.dataRoot)
        payload["outputRoot"] = str(self.outputRoot)
        return payload

    @classmethod
    def fromDict(cls, payload: dict) -> "SegmentationConfig":
        normalized = dict(payload)
        normalized["dataRoot"] = Path(normalized["dataRoot"])
        normalized["outputRoot"] = Path(normalized["outputRoot"])
        for tupleKey in ("patchSize", "spacing", "affineRotateRange", "affineScaleRange"):
            if tupleKey in normalized:
                normalized[tupleKey] = tuple(normalized[tupleKey])
        return cls(**normalized)
