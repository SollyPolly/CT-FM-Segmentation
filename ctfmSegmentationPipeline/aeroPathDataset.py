from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from ctfmSegmentationPipeline.config import SegmentationConfig
from ctfmSegmentationPipeline.dataTransforms import buildInferenceTransforms, buildTrainTransforms, buildValTransforms
from ctfmSegmentationPipeline.pathUtils import discoverCaseRecords, splitCaseRecords


def buildDataLoaders(
    config: SegmentationConfig,
) -> tuple[DataLoader, DataLoader, list[dict], list[dict]]:
    caseRecords = discoverCaseRecords(config.dataRoot)
    trainRecords, valRecords = splitCaseRecords(
        caseRecords=caseRecords,
        trainFraction=config.trainFraction,
        randomSeed=config.randomSeed,
    )

    datasetClass = CacheDataset if config.cacheRate > 0.0 else Dataset
    datasetKwargs = {"cache_rate": config.cacheRate} if config.cacheRate > 0.0 else {}

    trainDataset = datasetClass(
        data=trainRecords,
        transform=buildTrainTransforms(config),
        **datasetKwargs,
    )
    valDataset = datasetClass(
        data=valRecords,
        transform=buildValTransforms(config),
        **datasetKwargs,
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=config.trainBatchSize,
        shuffle=True,
        num_workers=config.numWorkers,
        collate_fn=list_data_collate,
        pin_memory=config.device == "cuda",
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=config.valBatchSize,
        shuffle=False,
        num_workers=config.numWorkers,
        pin_memory=config.device == "cuda",
    )
    return trainLoader, valLoader, trainRecords, valRecords


def buildInferenceDataset(config: SegmentationConfig, record: dict) -> Dataset:
    includeLungMask = bool(record.get(config.lungMaskKey))
    return Dataset(
        data=[record],
        transform=buildInferenceTransforms(config=config, includeLungMask=includeLungMask),
    )
