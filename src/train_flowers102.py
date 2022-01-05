from logging import INFO, basicConfig, getLogger

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomContrast,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

from arcface import AdditiveAngularMarginLoss
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


def main():
    set_gpu_memory_growth()

    dataset_id = "oxford_flowers102"
    _, metadata = tfds.load(dataset_id, with_info=True)
    n_classes = metadata.features["label"].num_classes
    logger.info(f"# of classes: {n_classes}")

    train_ds, valid_ds = tfds.load(
        dataset_id,
        split=["train", "validation"],
        as_supervised=True,
        decoders={"label": onehot_encoding(depth=n_classes)},
    )

    input_shape = (224, 224, 3)
    margin = 0.5
    scale = 64
    embedding_dimension = 64
    momentum = 0.9
    weights_decay = 5e-4
    batch_size = 32
    epochs = 100

    optimizer = tf.keras.optimizers.SGD(momentum=momentum)

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

    # random_state = 42
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(
                # [-50% * 2pi, 50% * 2pi] = 360 degree
                factor=0.05,
                fill_mode="nearest",
            ),
            # RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="wrap"),
            RandomZoom(height_factor=0.1, fill_mode="reflect"),
            RandomCrop(height=224, width=224),
            RandomContrast(factor=0.1),
        ]
    )

    train_data_size = len(train_ds)
    n_repeats = 10
    shuffle_buffer_size = train_data_size * n_repeats

    train_ds = (
        train_ds.map(
            lambda x, y: (preprocess_image(x, 256, 256), y), num_parallel_calls=AUTOTUNE
        )
        .cache()
        .repeat(n_repeats)
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(batch_size)
        .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    valid_ds = (
        valid_ds.map(
            lambda x, y: (preprocess_image(x, 224, 224), y), num_parallel_calls=AUTOTUNE
        )
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )

    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        embedding_dimension=embedding_dimension,
        weights_decay=weights_decay,
        use_pretrain=False,
    )

    model_path = "./model/weights.hdf5"

    # model_path_from = "./model/weights.hdf5.bkup"
    # if Path(model_path_from).exists():
    #     logger.info(f"load model weights from {model_path_from}")
    #     model.load_weights(model_path_from)

    model_checkpoint = ModelCheckpoint(
        # "./model/weights.{epoch:03d}-{val_loss:.3f}.hdf5",
        model_path,
        monitor="val_loss",
        save_best_only=True,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)

    model.compile(
        optimizer=optimizer,
        loss=AdditiveAngularMarginLoss(
            loss_func=tf.keras.losses.CategoricalCrossentropy(),
            margin=margin,
            scale=scale,
        ),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=[model_checkpoint, tensorboard_callback, lr_scheduler],
        verbose=2,
    )


if __name__ == "__main__":
    main()
