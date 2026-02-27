import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from ctfmSegmentationPipeline.aeroPathDataset import buildDataLoaders
from ctfmSegmentationPipeline.config import SegmentationConfig
from ctfmSegmentationPipeline.losses import buildLossFunction, computeBinaryDice
from ctfmSegmentationPipeline.pathUtils import ensureExperimentDirs, writeJson
from ctfmSegmentationPipeline.segmentationModel import (
    buildOptimizer,
    buildSegmentationModel,
    countTrainableParameters,
    setEncoderTrainable,
)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CT-FM initialized airway segmentation model.")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inference-overlap", type=float, default=0.5)
    parser.add_argument("--no-ctfm-init", action="store_true")
    parser.add_argument("--use-lung-roi", action="store_true")
    parser.add_argument("--normalize-orientation", action="store_true")
    parser.add_argument("--use-spacing", action="store_true")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"))
    parser.add_argument("--amp", action="store_true")
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
        numWorkers=args.num_workers,
        device=args.device,
        threshold=args.threshold,
        inferenceOverlap=args.inference_overlap,
        useCtfmWeights=not args.no_ctfm_init,
        useLungRoi=args.use_lung_roi,
        normalizeOrientation=args.normalize_orientation,
        useSpacing=args.use_spacing,
        spacing=tuple(args.spacing),
        useAmp=args.amp,
    )


def main() -> None:
    args = parseArgs()
    config = buildConfigFromArgs(args)
    setRandomSeeds(config.randomSeed)
    ensureExperimentDirs(config.experimentRoot())

    trainLoader, valLoader, trainRecords, valRecords = buildDataLoaders(config)
    model, loadSummary = buildSegmentationModel(config)
    model = model.to(config.device)

    optimizer = buildOptimizer(model, config)
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
        }

        if epochIndex % config.validateEvery == 0:
            valLoss, valDice = runValidationEpoch(
                model=model,
                valLoader=valLoader,
                lossFunction=lossFunction,
                config=config,
            )
            epochSummary["valLoss"] = valLoss
            epochSummary["valDice"] = valDice

            if valDice > bestValDice:
                bestValDice = valDice
                saveCheckpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epochIndex=epochIndex,
                    bestValDice=bestValDice,
                )

        history.append(epochSummary)
        print(json.dumps(epochSummary))

    writeJson({"history": history}, config.historyPath())
    writeJson(config.toDict(), config.configPath())
    print(f"Best validation dice: {bestValDice:.4f}")
    print(f"Saved config to: {config.configPath()}")


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
        scaler.step(optimizer)
        scaler.update()

        runningLoss += float(loss.item())
        batchCount += 1

    return runningLoss / max(batchCount, 1)


def runValidationEpoch(model, valLoader, lossFunction, config: SegmentationConfig) -> tuple[float, float]:
    model.eval()
    runningLoss = 0.0
    runningDice = 0.0
    caseCount = 0

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
            runningDice += computeBinaryDice(logits=logits, labels=labels, threshold=config.threshold)
            caseCount += 1

    return runningLoss / max(caseCount, 1), runningDice / max(caseCount, 1)


def saveCheckpoint(model, optimizer, config: SegmentationConfig, epochIndex: int, bestValDice: float) -> None:
    payload = {
        "epochIndex": epochIndex,
        "bestValDice": bestValDice,
        "config": config.toDict(),
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
    }
    torch.save(payload, config.checkpointPath())
    print(f"Saved checkpoint to: {config.checkpointPath()}")


def setRandomSeeds(randomSeed: int) -> None:
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randomSeed)


if __name__ == "__main__":
    main()
