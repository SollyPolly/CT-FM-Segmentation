import torch
from monai.losses import DiceCELoss
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric


def buildLossFunction() -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


class BinaryMetricTracker:
    def __init__(self, threshold: float, computeSurfaceMetrics: bool = True) -> None:
        self.threshold = threshold
        self.computeSurfaceMetrics = computeSurfaceMetrics

        self.metricSums = {
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
        }
        self.metricCounts = {key: 0 for key in self.metricSums}

        self.surfaceSums = {
            "hd95Vox": 0.0,
            "avgSurfaceDistVox": 0.0,
        }
        self.surfaceCounts = {key: 0 for key in self.surfaceSums}

        self.hd95Metric = None
        self.surfaceDistanceMetric = None
        if self.computeSurfaceMetrics:
            self.hd95Metric = HausdorffDistanceMetric(
                include_background=True,
                percentile=95,
                reduction="mean",
            )
            self.surfaceDistanceMetric = SurfaceDistanceMetric(
                include_background=True,
                reduction="mean",
                symmetric=True,
            )

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.threshold).float()
        labels = (labels > 0.5).float()

        batchMetrics = computeBatchOverlapMetrics(predictions=predictions, labels=labels)
        for key, value in batchMetrics.items():
            self.metricSums[key] += value
            self.metricCounts[key] += 1

        if not self.computeSurfaceMetrics:
            return

        if self.hd95Metric is None or self.surfaceDistanceMetric is None:
            return

        try:
            self.hd95Metric(y_pred=predictions, y=labels)
            hd95Value = self.hd95Metric.aggregate()
            self.hd95Metric.reset()
            self._accumulateSurfaceMetric("hd95Vox", hd95Value)
        except Exception:
            self.hd95Metric.reset()

        try:
            self.surfaceDistanceMetric(y_pred=predictions, y=labels)
            avgSurfaceDistValue = self.surfaceDistanceMetric.aggregate()
            self.surfaceDistanceMetric.reset()
            self._accumulateSurfaceMetric("avgSurfaceDistVox", avgSurfaceDistValue)
        except Exception:
            self.surfaceDistanceMetric.reset()

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, metricSum in self.metricSums.items():
            count = self.metricCounts[key]
            metrics[key] = metricSum / max(count, 1)

        for key, metricSum in self.surfaceSums.items():
            count = self.surfaceCounts[key]
            metrics[key] = metricSum / max(count, 1) if count > 0 else float("nan")

        return metrics

    def _accumulateSurfaceMetric(self, metricName: str, metricValue) -> None:
        scalarValue = _toFiniteFloat(metricValue)
        if scalarValue is None:
            return
        self.surfaceSums[metricName] += scalarValue
        self.surfaceCounts[metricName] += 1


def computeBatchOverlapMetrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    intersection = (predictions * labels).sum(dim=(1, 2, 3, 4))
    predPositives = predictions.sum(dim=(1, 2, 3, 4))
    labelPositives = labels.sum(dim=(1, 2, 3, 4))

    precision = (intersection + 1e-5) / (predPositives + 1e-5)
    recall = (intersection + 1e-5) / (labelPositives + 1e-5)
    dice = (2.0 * intersection + 1e-5) / (predPositives + labelPositives + 1e-5)
    union = predPositives + labelPositives - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)

    return {
        "dice": float(dice.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "iou": float(iou.mean().item()),
    }


def _toFiniteFloat(metricValue) -> float | None:
    if isinstance(metricValue, torch.Tensor):
        if metricValue.numel() == 0:
            return None
        scalar = float(metricValue.detach().cpu().mean().item())
    else:
        scalar = float(metricValue)

    if scalar != scalar or scalar == float("inf") or scalar == float("-inf"):
        return None
    return scalar
