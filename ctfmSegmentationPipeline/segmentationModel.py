import torch
from monai.networks.nets import SegResNet

from ctfmSegmentationPipeline.config import SegmentationConfig
from ctfmSegmentationPipeline.ctfmWeightLoader import WeightLoadSummary, loadCtfmEncoderWeights


def buildSegmentationModel(config: SegmentationConfig) -> tuple[SegResNet, WeightLoadSummary | None]:
    model = SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=1,
        act="relu",
        norm="batch",
        blocks_down=(1, 2, 2, 4, 4),
        blocks_up=(1, 1, 1, 1),
        use_conv_final=True,
    )

    loadSummary = None
    if config.useCtfmWeights:
        loadSummary = loadCtfmEncoderWeights(model=model, modelId=config.ctfmModelId)

    return model, loadSummary


def setEncoderTrainable(model: SegResNet, isTrainable: bool) -> None:
    for module in [model.convInit, model.down_layers]:
        for parameter in module.parameters():
            parameter.requires_grad = isTrainable


def buildOptimizer(model: SegResNet, config: SegmentationConfig) -> torch.optim.Optimizer:
    encoderParams = list(model.convInit.parameters()) + list(model.down_layers.parameters())
    decoderParams = list(model.up_layers.parameters()) + list(model.up_samples.parameters()) + list(model.conv_final.parameters())

    return torch.optim.AdamW(
        [
            {"params": encoderParams, "lr": config.encoderLearningRate},
            {"params": decoderParams, "lr": config.decoderLearningRate},
        ],
        weight_decay=config.weightDecay,
    )


def countTrainableParameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
