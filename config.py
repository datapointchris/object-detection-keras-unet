from dataclasses import dataclass, asdict, field
from pathlib import Path


class BaseConfig:
    def as_dict(self):
        return asdict(self)


@dataclass
class MainConfig(BaseConfig):
    SEED: int = 77
    loop_epochs: int = 5
    train_image_path: Path = Path('images/training/build/0/')
    train_mask_path: Path = Path('images/training/mask/0/')
    test_image_path: Path = Path('images/training/test/0/')
    s3_bucket: str = "s3://object-detection-keras-unet/"


@dataclass
class ImageProcessConfig(BaseConfig):
    split: bool = False
    split_rows: int = 200
    split_cols: int = 200
    resize: bool = True
    resize_width: int = 240
    resize_height: int = 240


@dataclass
class ModelConfig(BaseConfig):
    name: str = 'unet_object_detection'
    model_dir: Path = Path('models/')
    full_path: Path = model_dir / f'{name}.hdf5'
    train_val_split_size: float = 0.1
    pretrained: bool = False
    print_summary: bool = False
    plot_validation_images: bool = True
    data_augmentation: bool = False
    optimizer: str = 'adam'
    loss: str = 'binary_crossentropy'
    metrics: list = field(default_factory=lambda: ['accuracy'])


@dataclass
class ModelParams(BaseConfig):
    epochs: int = 5
    batch_size: int = 16
    verbose: int = 1


@dataclass
class DatagenParams(BaseConfig):
    featurewise_center: bool = (False,)  # set input mean to 0 over the dataset
    samplewise_center: bool = (False,)  # set each sample mean to 0
    featurewise_std_normalization: bool = (False,)  # divide inputs by std of the dataset
    samplewise_std_normalization: bool = (False,)  # divide each input by its std
    zca_whitening: bool = (False,)  # apply ZCA whitening
    zca_epsilon: float = (1e-06,)  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range = (60,)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range = (0.2,)
    # randomly shift images vertically (fraction of total height)
    height_shift_range = (0.2,)
    shear_range = (0.0,)  # set range for random shear
    zoom_range = (0.0,)  # set range for random zoom
    channel_shift_range = (0.0,)  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode = ('nearest',)
    cval = (0.0,)  # value used for fill_mode = "constant"
    horizontal_flip = (True,)  # randomly flip images
    vertical_flip = (True,)  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale = (None,)
    # set function that will be applied on each input
    preprocessing_function = (None,)
    # image data format, either "channels_first" or "channels_last"
    data_format = ('channels_last',)
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split = (0.0,)


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
datagen_params = DatagenParams()
tensorboard = TensorBoardParams()
