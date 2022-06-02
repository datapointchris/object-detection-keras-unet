import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import islice


def image_splitter(image, num_col_splits, num_row_splits):
    """
    Splits an image into 'num_col_splits' X 'num_row_splits'
    Returns array of images arranged from left -> right, top -> bottom
    """

    width = image.shape[0]
    height = image.shape[1]
    imglist = []

    for startpoint in np.linspace(0, width, num_col_splits, endpoint=False):
        endpoint = startpoint + (width / num_col_splits)

        for startp2 in np.linspace(0, height, num_row_splits, endpoint=False):
            endp2 = startp2 + (height / num_row_splits)

            imglist.append(image[int(startp2) : int(endp2), int(startpoint) : int(endpoint)])
    return np.array(imglist)


def print_array_properties(name, array):
    print('==========')
    print(name.upper())
    print(f'Length: {len(array)}')
    print(f'Shape: {array.shape}')
    print(f'Size: {round(array.itemsize * array.size / 1024 / 1024 / 1024, 3)} GB')
    print('==========')
    print('')


def create_batches(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def batch_stacker(batches, gray=False, resize: tuple[int] | None = None):
    for batch in batches:
        if gray:
            arr = [
                np.array(cv2.imread(image, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
                for image in batch
            ]
            if resize:
                arr = [cv2.resize(a, (resize[0], resize[1])) for a in arr]
            arr = np.expand_dims(arr, axis=3)
        else:
            arr = [np.array(cv2.imread(image), dtype=np.uint8) for image in batch]
            if resize:
                arr = [cv2.resize(a, (resize[0], resize[1])) for a in arr]
        stack = (np.stack(arr) / 255).astype(np.float32)
        yield stack


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
