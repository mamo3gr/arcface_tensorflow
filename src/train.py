from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import (
    CenterCrop,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)

from arcface import AdditiveAngularMarginLoss
from config_loader import load_setting
from losses import ClippedValueLoss
from model import create_model
from utils import set_gpu_memory_growth

basicConfig(level=INFO)
logger = getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_image(img, target_height, target_width):
    img = tf.image.resize(img, (target_height, target_width))
    img = preprocess_input(img)
    return img


@tfds.decode.make_decoder()
def onehot_encoding(example, feature, depth):
    return tf.one_hot(example, depth=depth, dtype=tf.int32)


def main(
    root_dir: str,
    split: str,
    input_shape: Tuple[int, int, int],
    n_classes: int,
    margin: float,
    scale: float,
    embedding_dimension: int,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    model_path: str,
    **kwargs,
):
    set_gpu_memory_growth()

    read_config = tfds.ReadConfig(shuffle_seed=seed)
    builder = tfds.ImageFolder(root_dir)
    ds = builder.as_dataset(
        split=split,
        batch_size=batch_size,
        shuffle_files=True,
        decoders={"label": onehot_encoding(depth=n_classes)},
        read_config=read_config,
        as_supervised=True,
    )

    height, width, n_channels = input_shape
    data_augmentation = tf.keras.Sequential(
        [
            RandomRotation(factor=0.05, fill_mode="nearest", seed=seed),
            RandomTranslation(
                height_factor=0.1, width_factor=0.1, fill_mode="wrap", seed=seed
            ),
            RandomZoom(height_factor=0.1, fill_mode="reflect", seed=seed),
            RandomContrast(factor=0.1, seed=seed),
            CenterCrop(height=height, width=width),
        ]
    )

    ds: tf.data.Dataset = (
        ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
        .unbatch()
    )

    valid_size = 1000
    valid_ds = ds.take(valid_size).batch(batch_size).prefetch(AUTOTUNE)
    train_ds = (
        ds.skip(valid_size)
        .shuffle(buffer_size=100000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        embedding_dimension=embedding_dimension,
        weights_decay=weight_decay,
        use_pretrain=False,
    )

    optimizer = tf.keras.optimizers.SGD(momentum=momentum)

    model_checkpoint = ModelCheckpoint(
        # "./model/weights.{epoch:03d}-{val_loss:.3f}.hdf5",
        model_path,
        monitor="val_loss",
        save_best_only=True,
    )

    def scheduler(epoch, lr):
        if epoch < 30:
            return 1e-1
        elif epoch < 60:
            return 1e-2
        elif epoch < 90:
            return 1e-3
        else:
            return 1e-4

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)

    model.compile(
        optimizer=optimizer,
        loss=ClippedValueLoss(
            loss_func=AdditiveAngularMarginLoss(
                loss_func=tf.keras.losses.CategoricalCrossentropy(),
                margin=margin,
                scale=scale,
            ),
            x_min=tf.keras.backend.epsilon(),
            x_max=1.0,
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=[
            model_checkpoint,
            lr_scheduler,
            tensorboard_callback,
        ],
        verbose=1,
    )


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_setting(args.config)
    main(**config)
