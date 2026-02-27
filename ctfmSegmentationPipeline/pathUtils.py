import json
import random
from pathlib import Path


def discoverCaseRecords(dataRoot: Path) -> list[dict]:
    caseRecords: list[dict] = []
    for caseDir in sorted([path for path in dataRoot.iterdir() if path.is_dir()], key=_caseSortKey):
        caseId = caseDir.name
        imagePath = caseDir / f"{caseId}_CT_HR.nii.gz"
        labelPath = caseDir / f"{caseId}_CT_HR_label_airways.nii.gz"
        lungMaskPath = caseDir / f"{caseId}_CT_HR_label_lungs.nii.gz"

        if not imagePath.exists() or not labelPath.exists():
            continue

        caseRecords.append(
            {
                "caseId": caseId,
                "image": str(imagePath),
                "label": str(labelPath),
                "lungMask": str(lungMaskPath) if lungMaskPath.exists() else None,
            }
        )

    if not caseRecords:
        raise FileNotFoundError(f"No AeroPath cases were found under {dataRoot}.")
    return caseRecords


def splitCaseRecords(caseRecords: list[dict], trainFraction: float, randomSeed: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 < trainFraction < 1.0:
        raise ValueError("trainFraction must be in (0.0, 1.0).")

    shuffledRecords = list(caseRecords)
    random.Random(randomSeed).shuffle(shuffledRecords)
    splitIndex = max(1, min(len(shuffledRecords) - 1, int(round(len(shuffledRecords) * trainFraction))))
    trainRecords = sorted(shuffledRecords[:splitIndex], key=lambda record: _caseSortKey(Path(record["image"]).parent))
    valRecords = sorted(shuffledRecords[splitIndex:], key=lambda record: _caseSortKey(Path(record["image"]).parent))
    return trainRecords, valRecords


def ensureExperimentDirs(experimentRoot: Path) -> None:
    experimentRoot.mkdir(parents=True, exist_ok=True)
    (experimentRoot / "inference").mkdir(parents=True, exist_ok=True)
    (experimentRoot / "figures").mkdir(parents=True, exist_ok=True)


def writeJson(payload: dict, outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _caseSortKey(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name
