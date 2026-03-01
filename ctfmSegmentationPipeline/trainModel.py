import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from ctfmSegmentationPipeline.aeroPathDataset import buildDataLoaders, buildDataLoadersFromRecords
from ctfmSegmentationPipeline.config import SegmentationConfig
from ctfmSegmentationPipeline.losses import BinaryMetricTracker, buildLossFunction
from ctfmSegmentationPipeline.pathUtils import ensureExperimentDirs, writeJson
from ctfmSegmentationPipeline.segmentationModel import (
    buildOptimizer,
    buildSegmentationModel,
    countTrainableParameters,
    setEncoderTrainable,
)


def addTrainingArgs(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, default=Path("data/AeroPath"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ctfmSegmentation"))
    parser.add_argument("--experiment-name", type=str, default="baselineSegResNet")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96), metavar=("D", "H", "W"))
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--disable-surface-metrics", action="store_true")
    parser.add_argument("--inference-overlap", type=float, default=0.5)
    parser.add_argument("--affine-augment-probability", type=float, default=0.2)
    parser.add_argument("--affine-rotate-range", type=float, nargs=3, default=(0.1, 0.1, 0.1), metavar=("Z", "Y", "X"))
    parser.add_argument("--affine-scale-range", type=float, nargs=3, default=(0.1, 0.1, 0.1), metavar=("Z", "Y", "X"))
    parser.add_argument("--noise-augment-probability", type=float, default=0.15)
    parser.add_argument("--intensity-scale-augment-probability", type=float, default=0.15)
    parser.add_argument("--intensity-shift-augment-probability", type=float, default=0.15)
    parser.add_argument("--no-ctfm-init", action="store_true")
    parser.add_argument("--use-lung-roi", action="store_true")
    parser.add_argument("--normalize-orientation", action="store_true")
    parser.add_argument("--use-spacing", action="store_true")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"))
    parser.add_argument("--amp", action="store_true")
    return parser


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CT-FM initialized airway segmentation model.")
    addTrainingArgs(parser)
    return parser.parse_args()


def buildConfigFromArgs(args: argparse.Namespace) -> SegmentationConfig:
    return SegmentationConfig(
        dataRoot=args.data_root,
        outputRoot=args.output_root,
        experimentName=args.experiment_name,
        trainFraction=args.train_fraction,
        patchSize=tuple(args.patch_size),
        trainBatchSize=args.train_batch_size,
        samplesPerCase=args.samples_per_case,
        maxEpochs=args.max_epochs,
        freezeEncoderEpochs=args.freeze_encoder_epochs,
        encoderLearningRate=args.encoder_learning_rate,
        decoderLearningRate=args.decoder_learning_rate,
        weightDecay=args.weight_decay,
        gradClipNorm=args.grad_clip_norm,
        earlyStoppingPatience=args.early_stopping_patience,
        earlyStoppingMinDelta=args.early_stopping_min_delta,
        lrSchedulerFactor=args.lr_scheduler_factor,
        lrSchedulerPatience=args.lr_scheduler_patience,
        minLearningRate=args.min_learning_rate,
        numWorkers=args.num_workers,
        device=args.device,
        threshold=args.threshold,
        computeSurfaceMetrics=not args.disable_surface_metrics,
        inferenceOverlap=args.inference_overlap,
        affineAugmentProbability=args.affine_augment_probability,
        affineRotateRange=tuple(args.affine_rotate_range),
        affineScaleRange=tuple(args.affine_scale_range),
        noiseAugmentProbability=args.noise_augment_probability,
        intensityScaleAugmentProbability=args.intensity_scale_augment_probability,
        intensityShiftAugmentProbability=args.intensity_shift_augment_probability,
        useCtfmWeights=not args.no_ctfm_init,
        useLungRoi=args.use_lung_roi,
        normalizeOrientation=args.normalize_orientation,
        useSpacing=args.use_spacing,
        spacing=tuple(args.spacing),
        useAmp=args.amp,
    )


def main() -> None:
    configureRuntimeWarnings()
    args = parseArgs()
    config = buildConfigFromArgs(args)
    runTraining(config=config)


def runTraining(
    config: SegmentationConfig,
    trainRecords: list[dict] | None = None,
    valRecords: list[dict] | None = None,
) -> dict:
    setRandomSeeds(config.randomSeed)
    ensureExperimentDirs(config.experimentRoot())

    if trainRecords is None or valRecords is None:
        trainLoader, valLoader, trainRecords, valRecords = buildDataLoaders(config)
    else:
        trainLoader, valLoader, trainRecords, valRecords = buildDataLoadersFromRecords(
            config=config,
            trainRecords=trainRecords,
            valRecords=valRecords,
        )
    model, loadSummary = buildSegmentationModel(config)
    model = model.to(config.device)

    optimizer = buildOptimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=config.lrSchedulerFactor,
        patience=config.lrSchedulerPatience,
        min_lr=config.minLearningRate,
    )
    lossFunction = buildLossFunction()
    scaler = torch.amp.GradScaler("cuda", enabled=config.useAmp and config.device == "cuda")

    if config.freezeEncoderEpochs > 0:
        setEncoderTrainable(model, isTrainable=False)

    print(f"Training cases: {len(trainRecords)}")
    print(f"Validation cases: {len(valRecords)}")
    print(f"Trainable parameters at start: {countTrainableParameters(model):,}")
    if loadSummary is not None:
        print(
            "Loaded CT-FM encoder weights: "
            f"{loadSummary.loadedKeys}/{loadSummary.sourceKeyCount} source tensors matched"
        )

    history: list[dict] = []
    bestValDice = -1.0
    bestEpochIndex = 0
    bestValMetrics: dict[str, float] = {}
    noImproveValidationCount = 0

    for epochIndex in range(1, config.maxEpochs + 1):
        if epochIndex == config.freezeEncoderEpochs + 1:
            setEncoderTrainable(model, isTrainable=True)
            print(f"Unfroze encoder at epoch {epochIndex}")
            print(f"Trainable parameters now: {countTrainableParameters(model):,}")

        trainLoss = runTrainEpoch(
            model=model,
            trainLoader=trainLoader,
            optimizer=optimizer,
            lossFunction=lossFunction,
            scaler=scaler,
            config=config,
        )

        epochSummary = {
            "epoch": epochIndex,
            "trainLoss": trainLoss,
            "encoderLr": optimizer.param_groups[0]["lr"],
            "decoderLr": optimizer.param_groups[1]["lr"],
        }

        if epochIndex % config.validateEvery == 0:
            valLoss, valMetrics = runValidationEpoch(
                model=model,
                valLoader=valLoader,
                lossFunction=lossFunction,
                config=config,
            )
            valDice = valMetrics["dice"]
            epochSummary["valLoss"] = valLoss
            epochSummary["valDice"] = valDice
            epochSummary["valPrecision"] = valMetrics["precision"]
            epochSummary["valRecall"] = valMetrics["recall"]
            epochSummary["valIoU"] = valMetrics["iou"]
            epochSummary["valHd95Vox"] = valMetrics["hd95Vox"]
            epochSummary["valAvgSurfaceDistVox"] = valMetrics["avgSurfaceDistVox"]
            scheduler.step(valDice)

            if valDice > bestValDice + config.earlyStoppingMinDelta:
                bestValDice = valDice
                bestEpochIndex = epochIndex
                bestValMetrics = {
                    "valLoss": valLoss,
                    "valDice": valDice,
                    "valPrecision": valMetrics["precision"],
                    "valRecall": valMetrics["recall"],
                    "valIoU": valMetrics["iou"],
                    "valHd95Vox": valMetrics["hd95Vox"],
                    "valAvgSurfaceDistVox": valMetrics["avgSurfaceDistVox"],
                }
                noImproveValidationCount = 0
                saveCheckpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epochIndex=epochIndex,
                    bestValDice=bestValDice,
                    outputPath=config.checkpointPath(),
                )
            else:
                noImproveValidationCount += 1

        history.append(epochSummary)
        saveCheckpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epochIndex=epochIndex,
            bestValDice=bestValDice,
            outputPath=config.lastCheckpointPath(),
            includeMessage=False,
        )
        writeJson({"history": history}, config.historyPath())
        print(json.dumps(epochSummary))

        if noImproveValidationCount >= config.earlyStoppingPatience:
            print(
                "Early stopping triggered after "
                f"{noImproveValidationCount} validation rounds without improvement."
            )
            break

    writeJson(config.toDict(), config.configPath())
    print(f"Best validation dice: {bestValDice:.4f}")
    print(f"Best validation epoch: {bestEpochIndex}")
    print(f"Saved config to: {config.configPath()}")
    return {
        "bestValDice": bestValDice,
        "bestEpochIndex": bestEpochIndex,
        "bestValMetrics": bestValMetrics,
        "historyPath": str(config.historyPath()),
        "checkpointPath": str(config.checkpointPath()),
        "trainCaseIds": [record["caseId"] for record in trainRecords],
        "valCaseIds": [record["caseId"] for record in valRecords],
    }


def runTrainEpoch(model, trainLoader, optimizer, lossFunction, scaler, config: SegmentationConfig) -> float:
    model.train()
    runningLoss = 0.0
    batchCount = 0

    for batch in tqdm(trainLoader, desc="train", leave=False):
        images = batch[config.imageKey].to(config.device)
        labels = batch[config.labelKey].float().to(config.device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=config.useAmp and config.device == "cuda"):
            logits = model(images)
            loss = lossFunction(logits, labels)

        scaler.scale(loss).backward()
        if config.gradClipNorm is not None and config.gradClipNorm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradClipNorm)
        scaler.step(optimizer)
        scaler.update()

        runningLoss += float(loss.item())
        batchCount += 1

    return runningLoss / max(batchCount, 1)


def runValidationEpoch(model, valLoader, lossFunction, config: SegmentationConfig) -> tuple[float, dict[str, float]]:
    model.eval()
    runningLoss = 0.0
    caseCount = 0
    metricTracker = BinaryMetricTracker(
        threshold=config.threshold,
        computeSurfaceMetrics=config.computeSurfaceMetrics,
    )

    with torch.inference_mode():
        for batch in tqdm(valLoader, desc="val", leave=False):
            images = batch[config.imageKey].to(config.device)
            labels = batch[config.labelKey].float().to(config.device)

            with torch.amp.autocast("cuda", enabled=config.useAmp and config.device == "cuda"):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=config.patchSize,
                    sw_batch_size=config.inferenceBatchSize,
                    predictor=model,
                    overlap=config.inferenceOverlap,
                )
                loss = lossFunction(logits, labels)

            runningLoss += float(loss.item())
            metricTracker.update(logits=logits, labels=labels)
            caseCount += 1

    return runningLoss / max(caseCount, 1), metricTracker.compute()


def saveCheckpoint(
    model,
    optimizer,
    config: SegmentationConfig,
    epochIndex: int,
    bestValDice: float,
    outputPath: Path,
    includeMessage: bool = True,
) -> None:
    payload = {
        "epochIndex": epochIndex,
        "bestValDice": bestValDice,
        "config": config.toDict(),
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
    }
    torch.save(payload, outputPath)
    if includeMessage:
        print(f"Saved checkpoint to: {outputPath}")


def setRandomSeeds(randomSeed: int) -> None:
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randomSeed)


def configureRuntimeWarnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The cuda.cudart module is deprecated.*",
        category=FutureWarning,
    )


if __name__ == "__main__":
    main()
