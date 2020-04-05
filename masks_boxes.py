#########################################################################
# CURRENTLY WORKING WITH TENSORFLOW
# 
# PRIORITIES:
# 1. refactor code to better groupings
# 2. where do the functions go
# 3. grab validation data
#
#
# If you are a potential EMPLOYER looking over my code, THANK YOU!
# I love to hear comments and critiques about being a better programmer,
# data engineer, and data scientist.  
# If you found this code useful or learned something, YAY!
# 
# LinkedIn: /in/chris-birch
# Portfolio: www.datapointchris.com/portfolio
# Website: www.datapointchris.com
#
#########################################################################














# # ================================================ FOR RUNNING ON THE MACBOOK PRO

# # DONT LEAVE COMMENTED CODE IN THE FINAL CODE. WOULD HAVE TO FIND OS IF NECESSARY BUT NOT

# ## DO THIS BEFORE IMPORTING KERAS OR TENSOR TO USE PLAIDML
# import plaidml.keras
# plaidml.keras.install_backend()

# # Help MacOS be able to use Keras
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# # Gets rid of the processor warning.
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Concatenate, Conv2DTranspose, UpSampling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from tqdm.keras import TqdmCallback
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import glob
import cv2
import os

import skimage.io


# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# =========================================================================Parameters, paths and variables
# hard coded for now

train_image_dir = '../images/network_training/build/0/'
train_mask_dir = '../images/network_training/mask/0/'
test_image_dir = '../images/network_training/test/0/'
data_dir = '../data/'

train_images = glob.glob(train_image_dir + '*')
train_filenames = [os.path.basename(x) for x in train_images]

test_images = glob.glob(test_image_dir + '*')
test_filenames = [os.path.basename(x) for x in test_images]


# decrease this number if running out of memory
# works with 4 on 11GB GPU
images_per_batch = 4

# val split from training set
train_val_split_size = .1

seed = 77


# IMAGE SPLITTER PARAMS #

split_rows = 20
split_cols = 20
resize = True
image_resize_width = 4800
image_resize_height = 4800


# MODEL PARAMS #

epochs = 5
batch_size = 1
pretrained_model = False
model_name = 'datagenmodel'
model_path = os.path.join(data_dir, model_name + '.h5')
plot_epoch_val_images = True
data_augmentation = False
early_stop = EarlyStopping(patience=5, verbose=1)
check_point = ModelCheckpoint(os.path.join(data_dir, model_name + '.hdf5'),
                              save_best_only=True,
                              verbose=1)

datagen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                             samplewise_center=False,  # set each sample mean to 0
                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                             samplewise_std_normalization=False,  # divide each input by its std
                             zca_whitening=False,  # apply ZCA whitening
                             zca_epsilon=1e-06,  # epsilon for ZCA whitening
                             # randomly rotate images in the range (degrees, 0 to 180)
                             rotation_range=0,
                             # randomly shift images horizontally (fraction of total width)
                             width_shift_range=0.1,
                             # randomly shift images vertically (fraction of total height)
                             height_shift_range=0.1,
                             shear_range=0.,  # set range for random shear
                             zoom_range=0.,  # set range for random zoom
                             channel_shift_range=0.,  # set range for random channel shifts
                             # set mode for filling points outside the input boundaries
                             fill_mode='nearest',
                             cval=0.,  # value used for fill_mode = "constant"
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=False,  # randomly flip images
                             # set rescaling factor (applied before any other transformation)
                             rescale=None,
                             # set function that will be applied on each input
                             preprocessing_function=None,
                             # image data format, either "channels_first" or "channels_last"
                             data_format='channels_last',
                             # fraction of images reserved for validation (strictly between 0 and 1)
                             validation_split=0.0)


class ValPlotCallback(Callback):

    def on_train_end(self, logs=None):
        print('VALIDATION IMAGES')
        X_val_pred = model.predict(x_val, verbose=1, batch_size=batch_size)
        X_val_pred_mask = (x_val_pred > 0.5).astype(np.uint8)
        plot_predictions(original=x_val,
                         predicted=x_val_pred,
                         predicted_mask=x_val_pred_mask,
                         ground_truth=y_val,
                         repeat=True)


def load_image_as_array(image_name, image_dir, gray=False, resize=False):
    """
    Loads and splits an image
    Returns numpy array
    """
    if gray is False:
        image = cv2.imread(image_dir + image_name).astype(np.uint8)
    else:
        image = cv2.imread(image_dir + image_name, 0).astype(np.uint8)

    image_as_array = image_splitter(image,
                                 num_col_splits=split_cols,
                                 num_row_splits=split_rows,
                                 resize=resize,
                                 resize_height=image_resize_height,
                                 resize_width=imgage_resize_width)

    return image_as_array


def shape_and_mem(array):
    """Prints the shape and memory size of an array"""
    print(f'Shape: {array.shape}')
    print(f'Size: {round(array.itemsize * array.size / 1024 / 1024 / 1024, 3)} GB')

def image_splitter(image, num_col_splits, num_row_splits, resize=False, resize_width=None, resize_height=None):
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

    for startpoint in np.linspace(0,width,num_col_splits, endpoint=False):
        endpoint=startpoint + (width / num_col_splits)

        for startp2 in np.linspace(0,height,num_row_splits, endpoint=False):
            endp2=startp2 + (height / num_row_splits)

            imglist.append(image[int(startp2):int(endp2), int(startpoint):int(endpoint)]
                        )

    return np.array(imglist)

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

        plt.subplots_adjust(wspace=.3, hspace=.3)
        plt.show()


def make_model(model_name=model_name):
    '''
    Creates a new U-Net model
    '''
    inputs = Input((x_train.shape[1], x_train.shape[2], 3))
    s = Lambda(lambda x: x) (inputs)  # removed / 255

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    return model


def plot_predictions(original, predicted, predicted_mask, ground_truth=None, repeat=False):
    """
    Plots the original image, predicted image, mask from the predicted image, and ground truth mask.
    ground_truth: None for testing images without masks
    repeat: use the same first 4 images in the dataset for comparison
    """
    ncols_calc = 3
    if ground_truth is not None:
        ncols_calc = 4   
    
    for i in range(4):
        if repeat:
            ix = i
        else:
            ix = np.random.randint(0, predicted.shape[0])
            
        fig, ax = plt.subplots(ncols=ncols_calc, figsize=(ncols_calc*5, 24))

        ax[0].set_title('Original')
        ax[0].imshow(original[ix])
        ax[0].axis('off')

        ax[1].set_title('Predicted')
        ax[1].imshow(np.squeeze(predicted[ix]))
        ax[1].axis('off')    

        ax[2].set_title('Predicted Mask')
        ax[2].imshow(np.squeeze(predicted_mask[ix]))
        ax[2].axis('off')
        
        if ground_truth is not None:
            ax[3].set_title('Ground Truth')
            ax[3].imshow(np.squeeze(ground_truth[ix]))
            ax[3].axis('off')

        plt.subplots_adjust(wspace=.3, hspace=.3)
        plt.show()


# ========================================================== SPLIT AND BATCH IMAGES
# ========================================================== SPLIT AND BATCH THE IMAGES
# ========================================================== SPLIT AND BATCH THE IMAGES

# create batch sets of 'images_per_batch' size
training_sets = [train_filenames[i:i + images_per_batch]
                 for i in range(0, len(train_filenames), images_per_batch)]

for batch_number, train_set in enumerate(tqdm(training_sets), start=1):
    xlist = []
    ylist = []
    for name in tqdm(train_set):
        xlist.append(load_image_as_array(name, train_image_dir, resize=True))
        ylist.append(load_image_as_array(name, train_mask_dir, resize=True, gray=True))

    print('loadedimages')
    print('xlist')
    print(len(xlist))
    print(xlist[0].shape)
    print('ylist')
    print(len(ylist))
    print(ylist[0].shape)
    
    x = (np.vstack(xlist)/255).astype(np.float32)
    y = (np.vstack(ylist)/255).astype(np.float32)
    y = np.expand_dims(y, axis=3)  # grayscale
    
    print('stacked images')
    print('x and y')
    print(shape_and_mem(x))
    print(shape_and_mem(y))
    
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                           random_state=77,
                                           test_size=train_val_split_size)
    print('xtrain, etc')
    print(shape_and_mem(x_train))
    print(shape_and_mem(y_train))
    print(shape_and_mem(x_val))
    print(shape_and_mem(y_val))
                                                      
    if pretrained_model or batch_number > 1:
        print('Loading Trained Model')
        model = load_model(model_path)
    else:
        print('Creating New Model')
        model = make_model(pretrained_model=pretrained_model, model_name=model_name)

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_steps=(x_train.shape[0] // batch_size) * train_val_split_size,
                            callbacks=[early_stop, check_point, ValPlotCallback()])
    else:
        print('Using real-time data augmentation.')

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        image_datagen.fit(x_train, augment=True, seed=seed)
        mask_datagen.fit(y_train, augment=True, seed=seed)

        image_generator = image_datagen.flow(x_train,
                            class_mode=None,
                            seed=seed)

        mask_generator = mask_datagen.flow(y_train,
            class_mode=None,
            seed=seed)

        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)

        model.fit_generator(train_generator,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            verbose=1,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_steps=(x_train.shape[0] // batch_size) * train_val_split_size,
                            callbacks=[early_stop, check_point, ValPlotCallback()])

    model.save(model_path)


# model = load_model(model_path)


x_val_pred = model.predict(x_val, verbose=1, batch_size=batch_size)

model.evaluate(x=x_val, y=y_val, batch_size=batch_size)

# simple threshold to change to 1/0, mask
x_val_pred_mask = (x_val_pred > 0.5).astype(np.uint8)

plot_predictions(original=x_val, predicted=x_val_pred, predicted_mask=x_val_pred_mask, ground_truth=y_val)


X_test = [np.array(
        image_splitter(
            cv2.imread(TEST_IMAGE_DIR + img_name).astype(np.uint8), 
            num_col_splits=split_cols, 
            num_row_splits=split_rows,
            resize=True,
            resize_height=img_resize_height,
            resize_width=img_resize_width
        )
    ) for img_name in tqdm(test_filenames[:10])]

X_test = (np.vstack(X_test)/255).astype(np.float32)

shape_and_mem(X_test)



y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

shape_and_mem(y_pred)

y_pred_mask = (y_pred > 0.5).astype(np.uint8)

plot_predictions(original=X_test, predicted=X_test_pred, predicted_mask=X_test_pred_mask)





# result = cv2.bitwise_and(test_split[0], test_split[0], mask=prediction[0])

result
