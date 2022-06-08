from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_predictions(
    original, predicted, predicted_mask, ground_truth=None, repeat=False, num_validation_sets=1
):
    """
    Plots the original image, predicted image, mask from the predicted image, and ground truth mask.
    ground_truth: None for testing images without masks
    repeat: use the same first 4 images in the dataset for comparison
    """
    ncols_calc = 3 if ground_truth is None else 4

    for i in range(num_validation_sets):
        if repeat:
            ix = i
        else:
            ix = np.random.randint(0, predicted.shape[0])

        fig, ax = plt.subplots(ncols=ncols_calc, figsize=(ncols_calc * 5, 24))

        ax[0].set_title('Original')
        ax[0].imshow(original[ix])
        ax[0].axis('off')

        ax[1].set_title('Predicted')
        ax[1].imshow(cv2.cvtColor(np.squeeze(predicted[ix]), cv2.COLOR_BGR2RGB))
        ax[1].axis('off')

        ax[2].set_title('Predicted Mask')
        ax[2].imshow(cv2.cvtColor(np.squeeze(predicted_mask[ix]), cv2.COLOR_BGR2RGB))
        ax[2].axis('off')

        if ground_truth is not None:
            ax[3].set_title('Ground Truth')
            ax[3].imshow(cv2.cvtColor(np.squeeze(ground_truth[ix]), cv2.COLOR_BGR2RGB))
            ax[3].axis('off')

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()


class ValPlotCallback(Callback):
    def __init__(self, model, batch_size, x_val, y_val):
        self.model = model
        self.batch_size = batch_size
        self.x_val = x_val
        self.y_val = y_val

    def on_train_end(self, logs=None):
        print('VALIDATION IMAGES')
        x_val_pred = self.model.predict(self.x_val, verbose=1, batch_size=self.batch_size)
        x_val_pred_mask = (x_val_pred > 0.5).astype(np.uint8)
        plot_predictions(
            original=self.x_val,
            predicted=x_val_pred,
            predicted_mask=x_val_pred_mask,
            ground_truth=self.y_val,
            repeat=True,
        )
