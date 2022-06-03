# Using Satellite Images to Detect Buildings

## Project Overview

- Training the UNET architecture to detect building objects.
- Using generators to yield batches of images.
- Test out Kera's ImageDataGenerator
- Running on AWS Deep Learning AMI
- Custom Callback function that plots the validation images to visualize progress

## Project Structure

`models/` -- Saved model(s)

`callbacks.py` -- Custom validation image plotting callback

`helpers.py` -- Helper functions

`keras_unet_object_detection` -- Main file to run the model training

`mnist_test_install.py` -- Test installation

`model.py` -- UNETModel class

`images` -- Training and (optional) Augmented images
- Why is there the 0 in the name?
  - Something with the datagen...


## Setup

### Training on AWS

1. Select the AMI
   1. DL
2. Add S3 security policy to DL instance
3. Connect to instance
4. Git pull code
5. `Make sure .env is present and using `prod` settings
6. Set configuration settings in `config.py`
7. Run `train.py` script or `train.ipynb` notebook
   1. `train.py` will train and save the model to S3
   2. `train.ipynb` additionally includes the training validation images for monitoring



## Future Considerations

- There should be some way to do a 'gridsearch' of the hyperparameters such as batch size, epochs, resize, etc to find the best training model.
  - This is in lieu of making good judgement based on domain knowledge of CNN, UNET, and computer vision in general...
- LOGGING