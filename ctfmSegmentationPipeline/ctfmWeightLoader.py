from dataclasses import dataclass

from lighter_zoo import SegResEncoder as CtfmSegResEncoder


@dataclass
class WeightLoadSummary:
    loadedKeys: int
    skippedKeys: int
    sourceKeyCount: int
    targetKeyCount: int


def loadCtfmEncoderWeights(model, modelId: str) -> WeightLoadSummary:
    sourceModel = CtfmSegResEncoder.from_pretrained(modelId)
    sourceState = sourceModel.state_dict()
    targetState = model.state_dict()

    remappedState: dict[str, object] = {}
    _maybeMap(sourceState, targetState, remappedState, "conv_init.weight", "convInit.conv.weight")

    stageIndex = 0
    while f"layers.{stageIndex}.blocks.0.conv1.weight" in sourceState:
        blockIndex = 0
        while f"layers.{stageIndex}.blocks.{blockIndex}.conv1.weight" in sourceState:
            targetBlockIndex = blockIndex + 1
            sourcePrefix = f"layers.{stageIndex}.blocks.{blockIndex}"
            targetPrefix = f"down_layers.{stageIndex}.{targetBlockIndex}"

            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm1.weight", f"{targetPrefix}.norm1.weight")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm1.bias", f"{targetPrefix}.norm1.bias")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm1.running_mean", f"{targetPrefix}.norm1.running_mean")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm1.running_var", f"{targetPrefix}.norm1.running_var")
            _maybeMap(
                sourceState,
                targetState,
                remappedState,
                f"{sourcePrefix}.norm1.num_batches_tracked",
                f"{targetPrefix}.norm1.num_batches_tracked",
            )
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.conv1.weight", f"{targetPrefix}.conv1.conv.weight")

            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm2.weight", f"{targetPrefix}.norm2.weight")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm2.bias", f"{targetPrefix}.norm2.bias")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm2.running_mean", f"{targetPrefix}.norm2.running_mean")
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.norm2.running_var", f"{targetPrefix}.norm2.running_var")
            _maybeMap(
                sourceState,
                targetState,
                remappedState,
                f"{sourcePrefix}.norm2.num_batches_tracked",
                f"{targetPrefix}.norm2.num_batches_tracked",
            )
            _maybeMap(sourceState, targetState, remappedState, f"{sourcePrefix}.conv2.weight", f"{targetPrefix}.conv2.conv.weight")
            blockIndex += 1

        _maybeMap(
            sourceState,
            targetState,
            remappedState,
            f"layers.{stageIndex}.downsample.weight",
            f"down_layers.{stageIndex + 1}.0.conv.weight",
        )
        stageIndex += 1

    updatedState = dict(targetState)
    updatedState.update(remappedState)
    model.load_state_dict(updatedState, strict=False)

    return WeightLoadSummary(
        loadedKeys=len(remappedState),
        skippedKeys=len(sourceState) - len(remappedState),
        sourceKeyCount=len(sourceState),
        targetKeyCount=len(targetState),
    )


def _maybeMap(sourceState: dict, targetState: dict, remappedState: dict, sourceKey: str, targetKey: str) -> None:
    if sourceKey not in sourceState or targetKey not in targetState:
        return
    if tuple(sourceState[sourceKey].shape) != tuple(targetState[targetKey].shape):
        return
    remappedState[targetKey] = sourceState[sourceKey]
