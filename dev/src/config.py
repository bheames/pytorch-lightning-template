import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class Config:
    data_dir: pathlib.Path = pathlib.Path(__file__).parents[2] / "data"
    batch_size: int = 4
    max_epochs: int = 3
    log_every_n_steps: int = 2
    limit_train_batches: float | None = None

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
