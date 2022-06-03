import os
import random
from pathlib import Path

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm

import config
from callbacks import ValPlotCallback
from helpers import (
    batch_stacker,
    create_batches,
    image_checker,
    image_splitter,
    image_tester,
    print_array_properties,
)
from model import UNetModel

train_image_path = config.main.train_image_path
mask_image_path = config.main.train_mask_path

filenames = [f.name for f in train_image_path.glob('*.tif')]
random.shuffle(filenames)  # does this help training?

train_files = [str(train_image_path / name) for name in filenames]
mask_files = [str(mask_image_path / name) for name in filenames]

train_img_batches = create_batches(train_files, batch_size=config.model_params.batch_size)
train_mask_batches = create_batches(mask_files, batch_size=config.model_params.batch_size)

train_stack = batch_stacker(
    train_img_batches, resize=(config.image.resize_width, config.image.resize_height)
)
mask_stack = batch_stacker(
    train_mask_batches, resize=(config.image.resize_width, config.image.resize_height), gray=True
)

for batch_number, (train_batch, mask_batch) in enumerate(zip(train_stack, mask_stack), start=1):
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

    inputs = Input(shape=(x_train.shape[1:]))

    if config.model.pretrained or batch_number > 1:
        print('Loading Trained Model')
        model = load_model(config.model.model_full_path)
    else:
        print('Creating New Model')
        model = make_model(
            inputs=inputs, model_name=config.model.name, print_summary=config.model.print_summary
        )

    early_stop = EarlyStopping(patience=5, verbose=1)
    check_point = ModelCheckpoint(config.model.full_path, save_best_only=True, verbose=1)
    tensor_board = TensorBoard(**config.tensorboard.as_dict())
    validation_plots = ValPlotCallback(
        model=model, batch_size=config.model_params.batch_size, x_val=x_val, y_val=y_val
    )

    model_fit_params = dict(
        validation_data=(x_val, y_val),
        steps_per_epoch=max(x_train.shape[0] // config.model_params.batch_size, 1),
        validation_steps=max(
            (x_train.shape[0] // config.model_params.batch_size) * config.model.train_val_split_size,
            1,
        ),
        callbacks=[early_stop, check_point, tensor_board, validation_plots],
    )
    if config.model.data_augmentation:

        image_datagen = ImageDataGenerator(**config.datagen_params.as_dict())
        mask_datagen = ImageDataGenerator(**config.datagen_params.as_dict())

        image_datagen.fit(x_train, augment=True, seed=config.main.SEED)
        mask_datagen.fit(y_train, augment=True, seed=config.main.SEED)

        image_generator = image_datagen.flow(x_train, seed=config.main.SEED)
        mask_generator = mask_datagen.flow(y_train, seed=config.main.SEED)

        train_generator = zip(image_generator, mask_generator)

        model.fit(train_generator, {**config.model_params.as_dict(), **model_fit_params})
    else:
        model.fit(x_train, y_train, {**config.model_params.as_dict(), **model_fit_params})

        if batch_number == 1:  # to prevent loading an old model on second epoch
            model.save(config.model.full_path)
