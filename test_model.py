import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config
from callbacks import plot_predictions
from helpers import print_array_properties

model = load_model(config.model.full_path)

x_test = (np.vstack(x_test) / 255).astype(np.float32)

print_array_properties(x_test)

y_pred = model.predict(test_generator, verbose=1)

print_array_properties(y_pred)

y_pred_mask = (y_pred > 0.5).astype(np.uint8)

plot_predictions(original=x_test, predicted=y_pred, predicted_mask=y_pred_mask)


# https://www.jeremyjordan.me/evaluating-image-segmentation-models/

# result = cv2.bitwise_and(test_split[0], test_split[0], mask=prediction[0])

# result

# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

# https://www.jeremyjordan.me/evaluating-image-segmentation-models/
