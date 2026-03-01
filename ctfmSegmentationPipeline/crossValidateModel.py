import argparse
import random
from dataclasses import replace

import numpy as np

from ctfmSegmentationPipeline.pathUtils import discoverCaseRecords, writeJson
from ctfmSegmentationPipeline.trainModel import addTrainingArgs, buildConfigFromArgs, configureRuntimeWarnings, runTraining


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run k-fold cross-validation with a reserved holdout split for CT-FM airway segmentation."
    )
    addTrainingArgs(parser)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--summary-dir-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    configureRuntimeWarnings()
    args = parseArgs()
    config = buildConfigFromArgs(args)

    caseRecords = discoverCaseRecords(config.dataRoot)
    developmentRecords, holdoutRecords = splitDevelopmentAndHoldout(
        caseRecords=caseRecords,
        holdoutFraction=args.holdout_fraction,
        randomSeed=config.randomSeed,
    )
    foldSplits = buildFoldSplits(
        developmentRecords=developmentRecords,
        numFolds=args.num_folds,
        randomSeed=config.randomSeed,
    )

    summaryDirName = args.summary_dir_name or f"{config.experimentName}_crossVal"
    summaryDir = config.outputRoot / summaryDirName
    summaryDir.mkdir(parents=True, exist_ok=True)

    splitManifest = {
        "dataRoot": str(config.dataRoot),
        "numCases": len(caseRecords),
        "numDevelopmentCases": len(developmentRecords),
        "numHoldoutCases": len(holdoutRecords),
        "holdoutCaseIds": [record["caseId"] for record in holdoutRecords],
        "numFolds": args.num_folds,
    }
    writeJson(splitManifest, summaryDir / "splitManifest.json")

    foldResults: list[dict] = []
    for foldIndex, (foldTrainRecords, foldValRecords) in enumerate(foldSplits, start=1):
        foldConfig = replace(
            config,
            experimentName=f"{config.experimentName}_fold{foldIndex}",
        )
        print(
            f"Fold {foldIndex}/{args.num_folds}: "
            f"train={len(foldTrainRecords)} cases, val={len(foldValRecords)} cases"
        )
        foldResult = runTraining(
            config=foldConfig,
            trainRecords=foldTrainRecords,
            valRecords=foldValRecords,
        )
        foldResult["foldIndex"] = foldIndex
        foldResults.append(foldResult)
        writeJson({"foldResults": foldResults}, summaryDir / "foldResults.partial.json")

    aggregateSummary = aggregateFoldMetrics(foldResults)
    outputSummary = {
        "experimentName": config.experimentName,
        "numFolds": args.num_folds,
        "numDevelopmentCases": len(developmentRecords),
        "numHoldoutCases": len(holdoutRecords),
        "holdoutCaseIds": [record["caseId"] for record in holdoutRecords],
        "foldResults": foldResults,
        "aggregate": aggregateSummary,
    }
    writeJson(outputSummary, summaryDir / "crossValidationSummary.json")

    print(f"Saved split manifest to: {summaryDir / 'splitManifest.json'}")
    print(f"Saved cross-validation summary to: {summaryDir / 'crossValidationSummary.json'}")
    print(
        "Reserved holdout cases were not used in cross-validation. "
        "Use them once for final locked-model evaluation."
    )


def splitDevelopmentAndHoldout(
    caseRecords: list[dict],
    holdoutFraction: float,
    randomSeed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < holdoutFraction < 1.0:
        raise ValueError("holdoutFraction must be in (0.0, 1.0).")

    shuffledRecords = list(caseRecords)
    random.Random(randomSeed).shuffle(shuffledRecords)

    holdoutCount = int(round(len(shuffledRecords) * holdoutFraction))
    holdoutCount = max(1, min(len(shuffledRecords) - 1, holdoutCount))

    holdoutRecords = sorted(shuffledRecords[:holdoutCount], key=_recordSortKey)
    developmentRecords = sorted(shuffledRecords[holdoutCount:], key=_recordSortKey)
    return developmentRecords, holdoutRecords


def buildFoldSplits(
    developmentRecords: list[dict],
    numFolds: int,
    randomSeed: int,
) -> list[tuple[list[dict], list[dict]]]:
    if numFolds < 2:
        raise ValueError("numFolds must be at least 2.")
    if len(developmentRecords) < numFolds:
        raise ValueError(
            f"numFolds={numFolds} is too high for {len(developmentRecords)} development cases."
        )

    shuffledRecords = list(developmentRecords)
    random.Random(randomSeed).shuffle(shuffledRecords)

    foldBins: list[list[dict]] = [[] for _ in range(numFolds)]
    for recordIndex, record in enumerate(shuffledRecords):
        foldBins[recordIndex % numFolds].append(record)

    foldSplits: list[tuple[list[dict], list[dict]]] = []
    for foldIndex in range(numFolds):
        valRecords = sorted(foldBins[foldIndex], key=_recordSortKey)
        trainRecords = sorted(
            [
                record
                for otherFoldIndex, foldRecords in enumerate(foldBins)
                if otherFoldIndex != foldIndex
                for record in foldRecords
            ],
            key=_recordSortKey,
        )
        foldSplits.append((trainRecords, valRecords))

    return foldSplits


def aggregateFoldMetrics(foldResults: list[dict]) -> dict:
    if not foldResults:
        return {}

    metricNames = sorted(
        {
            metricName
            for foldResult in foldResults
            for metricName in foldResult.get("bestValMetrics", {}).keys()
        }
    )
    aggregate: dict[str, float] = {}
    for metricName in metricNames:
        metricValues = [
            float(foldResult["bestValMetrics"].get(metricName, float("nan")))
            for foldResult in foldResults
        ]
        finiteValues = [value for value in metricValues if np.isfinite(value)]
        if not finiteValues:
            aggregate[f"{metricName}Mean"] = float("nan")
            aggregate[f"{metricName}Std"] = float("nan")
            continue
        aggregate[f"{metricName}Mean"] = float(np.mean(finiteValues))
        aggregate[f"{metricName}Std"] = float(np.std(finiteValues))

    bestDiceValues = [float(result.get("bestValDice", float("nan"))) for result in foldResults]
    finiteBestDiceValues = [value for value in bestDiceValues if np.isfinite(value)]
    if finiteBestDiceValues:
        aggregate["bestValDiceMean"] = float(np.mean(finiteBestDiceValues))
        aggregate["bestValDiceStd"] = float(np.std(finiteBestDiceValues))
    else:
        aggregate["bestValDiceMean"] = float("nan")
        aggregate["bestValDiceStd"] = float("nan")

    return aggregate


def _recordSortKey(record: dict) -> tuple[int, str]:
    caseId = str(record.get("caseId", ""))
    try:
        return int(caseId), caseId
    except ValueError:
        return 10**9, caseId


if __name__ == "__main__":
    main()
