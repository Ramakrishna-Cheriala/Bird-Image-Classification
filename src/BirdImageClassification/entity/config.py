from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_api: Path
    local_files_dir: Path
    unzip_dir: Path


@dataclass(frozen=True)
class ModelPreparationConfig:
    root_dir: Path
    saved_model_dir: Path
    input_shape: int
    include_top: bool
    weights: str
    classes: int
    learning_rate: float


@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    trained_model_dir: Path
    train_data: Path
    test_data: Path
    validation_data: Path
    epochs: int
    batch_size: int
    saved_model_dir: Path
    input_shape: int
