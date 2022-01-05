from pathlib import Path
from typing import Union

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def preprocess_image(img, target_height, target_width):
    img = tf.image.resize(img, (target_height, target_width))
    img = preprocess_input(img)
    return img


def get_dataset(root_dir: Union[str, Path]):
    root_dir = Path(root_dir)


if __name__ == "__main__":
    get_dataset()
