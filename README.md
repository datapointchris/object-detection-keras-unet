# Using Satellite Images to Identify Disaster Damage

### Project Overview
====================
- Training the UNET architecture to detect building objects.
- Using generators to yield batches of images.
- Test out Kera's ImageDataGenerator
- Running on AWS Deep Learning AMI
- Custom Callback function that plots the validation images to visualize progress

### Project Structure
=====================
`data/` -- Saved model(s)
`callbacks.py` -- Custom validation image plotting callback
`helpers.py` -- Helper functions
`keras_cnn_working_example_cifar10_dataset.ipynb` -- From Keras
`keras_unet_object_detection` -- Main file to run the model training
`mnist_test_install.py` -- Test installation
`mnist_test_notebook.ipynb` -- Test TF and Jupyter install
`model.py` - UNET model
`positive_and_negative_images.ipynb` -- (Not using) - using thresholds

