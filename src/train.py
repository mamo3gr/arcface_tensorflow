from logging import INFO, basicConfig, getLogger

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

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

    root_dir = "/home/mamo/datasets/ms1m_align_112/"
    split = "imgs"  # Subsplit API not yet supported for ImageFolder
    input_shape = (112, 112, 3)
    margin = 0.5
    scale = 64
    embedding_dimension = 512
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 256
    epochs = 120
    seed = 42

    builder = tfds.ImageFolder(root_dir)
    n_classes = builder.info.features["label"].num_classes
    logger.info(f"# of classes: {n_classes}")

    read_config = tfds.ReadConfig(shuffle_seed=seed)

    ds = builder.as_dataset(
        split=split,
        batch_size=batch_size,
        shuffle_files=True,
        decoders={"label": onehot_encoding(depth=n_classes)},
        read_config=read_config,
        as_supervised=True,
    )

    ds: tf.data.Dataset = ds.map(
        lambda x, y: (preprocess_image(x, 112, 112), y), num_parallel_calls=AUTOTUNE
    ).unbatch()

    valid_size = 1000
    # train_size = int(n_samples * 0.95)
    valid_ds = ds.take(valid_size).batch(batch_size).prefetch(AUTOTUNE)
    train_ds = (
        ds.skip(valid_size)
        .shuffle(buffer_size=10000)
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
        callbacks=[
            model_checkpoint,
            lr_scheduler,
            tensorboard_callback,
        ],
        verbose=2,
    )


if __name__ == "__main__":
    main()
