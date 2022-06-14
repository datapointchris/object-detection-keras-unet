from dataclasses import dataclass, asdict, field
from pathlib import Path


class BaseConfig:
    def as_dict(self):
        return asdict(self)


@dataclass
class MainConfig(BaseConfig):
    ENVIRONMENT: str = 'development'
    SEED: int = 77
    loop_epochs: int = 5
    s3_bucket: str = 'object-detection-north-virginia'
    train_image_path: Path = Path('images/training/build/0/')
    train_mask_path: Path = Path('images/training/mask/0/')
    test_image_path: Path = Path('images/training/test/0/')


@dataclass
class ImageProcessConfig(BaseConfig):
    image_batch_size: int = 8
    resize_width: int = 1024
    resize_height: int = 1024
    augment: bool = False
    split_rows: int = 2
    split_cols: int = 2


@dataclass
class ModelConfig(BaseConfig):
    name: str = 'unet_object_detection'
    model_dir: Path = Path('models/')
    full_path: Path = model_dir / f'{name}.hdf5'
    train_val_split_size: float = 0.1
    pretrained: bool = False
    print_summary: bool = False
    plot_validation_images: bool = True
    optimizer: str = 'adam'
    loss: str = 'binary_crossentropy'
    metrics: list = field(default_factory=lambda: ['accuracy'])


@dataclass
class ModelParams(BaseConfig):
    epochs: int = 2
    batch_size: int = 8
    verbose: int = 1


@dataclass
class TensorBoardParams(BaseConfig):
    log_dir: Path = Path('logs/tensorboard/')
    histogram_freq: int = 1
    write_graph: bool = True
    write_grads: bool = False
    write_images: bool = True
    embeddings_freq: int = 1
    update_freq: str = 'epoch'


main = MainConfig()
image = ImageProcessConfig()
model = ModelConfig()
model_params = ModelParams()
tensorboard = TensorBoardParams()
