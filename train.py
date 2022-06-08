from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import load_model
from sklearn.model_selection import train_test_split

import config
from callbacks import ValPlotCallback
from helpers import (
    generate_augmentations,
    generate_training_batches,
    print_array_properties,
)
from model import make_model


def main_training_loop(loop_epochs, show_batch_data_shapes=True, show_validation_plots=False):
    for loop_epoch in range(1, loop_epochs + 1):
        print(f'LOOP EPOCH: {loop_epoch} of {loop_epochs}')

        batch_params = dict(
            batch_size=config.image.image_batch_size,
            resize=(config.image.resize_width, config.image.resize_height),
            split=(config.image.split_cols, config.image.split_rows),
            augment=config.image.augment,
            seed=config.main.SEED,
        )
        augment_params = generate_augmentations(seed=config.main.SEED)
        train_batches = generate_training_batches(
            path=config.main.train_image_path, gray=False, sync=augment_params, **batch_params
        )
        mask_batches = generate_training_batches(
            path=config.main.train_mask_path, gray=True, sync=augment_params, **batch_params
        )

        for batch_number, (train_batch, mask_batch) in enumerate(
            zip(train_batches, mask_batches), start=1
        ):
            print(f'BATCH NUMBER {batch_number}')
            print('AUGMENT PARAMS')
            print(augment_params.keys())

            x_train, x_val, y_train, y_val = train_test_split(
                train_batch,
                mask_batch,
                random_state=config.main.SEED,
                test_size=config.model.train_val_split_size,
            )
            if batch_number == 1 and show_batch_data_shapes:
                print_array_properties('x_train', x_train)
                print_array_properties('y_train', y_train)
                print_array_properties('x_val', x_val)
                print_array_properties('y_val', y_val)

            inputs = Input(shape=(x_train.shape[1:]))

            if config.model.pretrained or batch_number > 1 or loop_epoch > 1:
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
