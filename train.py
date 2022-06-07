import random
from pathlib import Path

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import config
from callbacks import ValPlotCallback
from helpers import (
    batch_stacker,
    create_batches,
    print_array_properties,
)
from model import make_model


def load_image_datasets(images_path: Path, masks_path: Path):
    train_image_path = images_path
    mask_image_path = masks_path

    filenames = [f.name for f in train_image_path.glob('*.tif')]
    random.shuffle(filenames)

    train_files = [str(train_image_path / name) for name in filenames]
    mask_files = [str(mask_image_path / name) for name in filenames]

    return train_files, mask_files


def augment_images(x_train, y_train, datagen_params: config.DatagenParams, seed: int) -> tuple:
    image_datagen = ImageDataGenerator(**datagen_params.as_dict())
    mask_datagen = ImageDataGenerator(**datagen_params.as_dict())

    image_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(x_train, seed=seed)
    mask_generator = mask_datagen.flow(y_train, seed=seed)

    return zip(image_generator, mask_generator)


def main_training_loop(loop_epochs=1, augment_data=False, show_validation_plots=False):
    for loop_epoch in range(1, loop_epochs + 1):
        print(f'Training Loop Epoch: {loop_epoch} of {loop_epochs}')
        if loop_epoch == 1:
            augment_data = False  # do not augment the first loop of original images

        train_files, mask_files = load_image_datasets(
            images_path=config.main.train_image_path, masks_path=config.main.train_mask_path
        )

        train_img_batches = create_batches(train_files, batch_size=config.model_params.batch_size)
        train_mask_batches = create_batches(mask_files, batch_size=config.model_params.batch_size)

        train_stack = batch_stacker(
            train_img_batches, resize=(config.image.resize_width, config.image.resize_height)
        )
        mask_stack = batch_stacker(
            train_mask_batches,
            resize=(config.image.resize_width, config.image.resize_height),
            gray=True,
        )

        for batch_number, (train_batch, mask_batch) in enumerate(
            zip(train_stack, mask_stack), start=1
        ):
            print(f'BATCH NUMBER {batch_number}')

            x_train, x_val, y_train, y_val = train_test_split(
                train_batch,
                mask_batch,
                random_state=config.main.SEED,
                test_size=config.model.train_val_split_size,
            )
            if batch_number == 1:
                print_array_properties('x_train', x_train)
                print_array_properties('y_train', y_train)
                print_array_properties('x_val', x_val)
                print_array_properties('y_val', y_val)

            if augment_data:
                print('Using Data Augmentation')
                x_train, y_train = augment_data(
                    x_train, y_train, datagen_params=config.datagen_params, seed=config.main.SEED
                )

            inputs = Input(shape=(x_train.shape[1:]))

            if config.model.pretrained or batch_number > 1:
                print('Loading Trained Model')
                model = load_model(config.model.full_path)
            else:
                print('Creating New Model')
                model = make_model(inputs=inputs, name=config.model.name)
                model.compile(
                    optimizer=config.model.optimizer,
                    loss=config.model.loss,
                    metrics=config.model.metrics,
                )
                if config.model.print_summary:
                    print(model.summary())

            early_stop = EarlyStopping(patience=5, verbose=1)
            check_point = ModelCheckpoint(config.model.full_path, save_best_only=True, verbose=1)
            tensor_board = TensorBoard(**config.tensorboard.as_dict())
            validation_plots = ValPlotCallback(
                model=model, batch_size=config.model_params.batch_size, x_val=x_val, y_val=y_val
            )
            all_callbacks = [early_stop, check_point, tensor_board]
            if show_validation_plots:
                all_callbacks.append(validation_plots)
            model_fit_params = dict(
                validation_data=(x_val, y_val),
                steps_per_epoch=max(x_train.shape[0] // config.model_params.batch_size, 1),
                validation_steps=max(
                    (x_train.shape[0] // config.model_params.batch_size)
                    * config.model.train_val_split_size,
                    1,
                ),
                callbacks=all_callbacks,
            )

            model.fit(x=x_train, y=y_train, **{**config.model_params.as_dict(), **model_fit_params})

            if batch_number == 1:  # to prevent loading an old model on second epoch
                model.save(config.model.full_path)


if __name__ == '__main__':
    main_training_loop(loop_epochs=config.main.loop_epochs)
