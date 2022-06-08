import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import islice
from functools import partial
import tensorflow as tf
import random
from pathlib import Path


def load_dataset_path(path: Path, format='.tif', limit=None):
    paths = sorted(list(path.glob(f'*{format}')))
    return paths[:limit] if limit else paths


def create_batches(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def load_image(path, gray=False):
    if gray:
        return np.array(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    else:
        return np.array(cv2.imread(str(path)), dtype=np.uint8)


def resize_image(image, resize_width, resize_height):
    return cv2.resize(image, (resize_width, resize_height))


def generate_augmentations(seed=77) -> dict:
    seed = (seed, 0)  # to make tf seed happy
    transforms = {
        'brightness': partial(tf.image.stateless_random_brightness, max_delta=0.4, seed=seed),
        'contrast': partial(tf.image.stateless_random_contrast, lower=0.1, upper=0.5, seed=seed),
        'flip_horizontal': partial(tf.image.stateless_random_flip_left_right, seed=seed),
        'flip_vertical': partial(tf.image.stateless_random_flip_up_down, seed=seed),
        'hue': partial(tf.image.stateless_random_hue, max_delta=0.5, seed=seed),
        'saturation': partial(
            tf.image.stateless_random_saturation, lower=0.1, upper=0.5, seed=seed
        ),
    }
    transform_range = range(0, len(transforms))
    number_of_transforms = (random.choices(transform_range, weights=reversed(transform_range)))[0]
    keys = random.sample(sorted(transforms), k=number_of_transforms)
    print('AUGMENTATIONS RETURNED')
    print(keys)
    return {k: transforms[k] for k in keys}


def augment_image(image, transforms, gray=False):
    gray_keys = ['brightness', 'contrast', 'hue', 'saturation']
    if gray:
        transforms = {k: transforms[k] for k in transforms if k not in gray_keys}
    if transforms:
        new = image
        for name in transforms:
            transform = transforms.get(name)
            new = transform(new)
        return np.array(new)
    return np.array(image)


def split_image(image, col_splits, row_splits):
    """
    Splits an image into 'col_splits' X 'row_splits'
    Returns array of images arranged from left -> right, top -> bottom
    """

    width = image.shape[0]
    height = image.shape[1]
    imglist = []

    for vstart in np.linspace(0, width, col_splits, endpoint=False):
        vend = vstart + (width / col_splits)

        for hstart in np.linspace(0, height, row_splits, endpoint=False):
            hend = hstart + (height / row_splits)

            imglist.append(image[int(hstart) : int(hend), int(vstart) : int(vend)])
    return np.array(imglist)


def print_array_properties(name, array):
    print('==========')
    print(name.upper())
    print(f'Length: {len(array)}')
    print(f'Shape: {array.shape}')
    print(f'Size: {round(array.itemsize * array.size / 1024 / 1024 / 1024, 3)} GB')
    print('==========')
    print('')


def generate_training_batches(
    path,
    batch_size,
    resize: tuple[int] | None = None,
    split: tuple[int] | None = None,
    augment: bool = None,
    gray: bool = False,
    seed: int = 77,
    sync: list | None = None,
):
    batches = create_batches(load_dataset_path(path, limit=None), batch_size=batch_size)
    for batch in batches:
        batch = (load_image(image) for image in batch)
        if gray:
            batch = (
                np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=2) for image in batch
            )
        if resize:
            batch = (resize_image(image, resize[0], resize[1]) for image in batch)
            if gray:
                batch = (np.expand_dims(image, axis=2) for image in batch)
        if augment:
            print('INFO: Augmenting')
            if sync:
                batch = (augment_image(image, transforms=sync, gray=gray) for image in batch)
            else:
                transforms = generate_augmentations(seed=seed)
                batch = (augment_image(image, transforms=transforms, gray=gray) for image in batch)
        if split:
            batch = (split_image(image, split[0], split[1]) for image in batch)
        batch = np.stack(list(batch)) / 255
        batch = batch.astype(np.float32)
        if len(batch[0].shape) < 4:
            batch = np.expand_dims(batch, axis=4)
        batch = batch.reshape(-1, batch[0].shape[1], batch[0].shape[2], batch[0].shape[3])
        yield batch


def image_tester(original):
    """Shows original images in array"""
    for _ in range(4):
        ix = np.random.randint(0, original.shape[0])
        fig, ax = plt.subplots(figsize=(10, 24))

        ax.set_title('Original')
        ax.imshow(original[ix])
        ax.axis('off')

        plt.show()


def image_checker(original, ground_truth):
    """Shows original images in array with their ground truth masks"""
    for _ in range(4):
        ix = np.random.randint(0, original.shape[0])
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 24))

        ax1.set_title('Original')
        ax1.imshow(original[ix])
        ax1.axis('off')

        ax2.set_title('Ground Truth')
        ax2.imshow(np.squeeze(ground_truth[ix]))
        ax2.axis('off')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
