import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


def buildLossFunction() -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


def buildDiceMetric() -> DiceMetric:
    return DiceMetric(include_background=True, reduction="mean")


def computeBinaryDice(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    intersection = (predictions * labels).sum(dim=(1, 2, 3, 4))
    denominator = predictions.sum(dim=(1, 2, 3, 4)) + labels.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return float(dice.mean().item())
