import numpy as np
import cv2


def image_splitter(
    image, num_col_splits, num_row_splits, resize=False, resize_width=None, resize_height=None
):
    """
    Splits an image into 'num_col_splits' X 'num_row_splits'
    Resize by setting resize=True and specifying 'resize_width' and 'resize_height'
    Returns array of images arranged from left -> right, top -> bottom
    """
    if resize:
        image = cv2.resize(image, (resize_width, resize_height))

    width = image.shape[0]
    height = image.shape[1]
    imglist = []

    for startpoint in np.linspace(0, width, num_col_splits, endpoint=False):
        endpoint = startpoint + (width / num_col_splits)

        for startp2 in np.linspace(0, height, num_row_splits, endpoint=False):
            endp2 = startp2 + (height / num_row_splits)

            imglist.append(image[int(startp2) : int(endp2), int(startpoint) : int(endpoint)])
    return np.array(imglist)


def load_image_as_array(image_name, image_dir, gray=False, resize=False):
    """
    Loads and splits an image
    Returns numpy array
    """
    if gray is False:
        image = cv2.imread(image_dir + image_name).astype(np.uint8)
    else:
        image = cv2.imread(image_dir + image_name, 0).astype(np.uint8)

    image_as_array = image_splitter(
        image,
        num_col_splits=split_cols,
        num_row_splits=split_rows,
        resize=resize,
        resize_height=image_resize_height,
        resize_width=image_resize_width,
    )
    return image_as_array


def array_shape_and_mem_usage(array):
    """Prints the shape and memory size of an array"""
    print(f'Shape: {array.shape}')
    print(f'Size: {round(array.itemsize * array.size / 1024 / 1024 / 1024, 3)} GB')


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
