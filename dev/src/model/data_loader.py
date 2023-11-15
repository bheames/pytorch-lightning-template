import enum
import pathlib
from typing import Any, cast

import torch
import torchvision


class Mode(enum.Enum):
    TRAIN = enum.auto()
    VALID = enum.auto()
    TEST = enum.auto()


class DataLoader(torch.utils.data.DataLoader[Any]):
    def __init__(self, data_dir: pathlib.Path, mode: Mode, **kwargs: Any) -> None:
        super().__init__(_get_dataset(data_dir, mode), **kwargs)


def _get_dataset(
    data_dir: pathlib.Path, mode: Mode = Mode.TRAIN
) -> torch.utils.data.Dataset[Any]:
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=mode is mode.TRAIN.value,
    )
    return cast(torch.utils.data.Dataset[Any], dataset)
