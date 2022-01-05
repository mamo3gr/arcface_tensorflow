import os
from argparse import ArgumentParser

import bcolz
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from more_itertools import chunked
from tensorflow.keras.applications.efficientnet import preprocess_input

from model import create_model
from utils import set_gpu_memory_growth

AUTOTUNE = tf.data.AUTOTUNE


def get_pair(root, name):
    """
    References:
        https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo
    """
    carray = bcolz.carray(rootdir=os.path.join(root, name), mode="r")
    issame = np.load("{}/{}_list.npy".format(root, name))
    return carray, issame


def get_data(data_root):
    """
    References:
        https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo
    """
    vgg2_fp, vgg2_fp_issame = get_pair(data_root, "vgg2_fp")
    return vgg2_fp, vgg2_fp_issame


def get_data_lfw(data_root):
    """
    References:
        https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo
    """
    vgg2_fp, vgg2_fp_issame = get_pair(data_root, "lfw")
    return vgg2_fp, vgg2_fp_issame


def evaluate_model(model_path: str, dataset_dir: str, batch_size=64):
    set_gpu_memory_growth()

    vgg2_fp, vgg2_fp_is_same = get_data(dataset_dir)
    # swap axis so as to convert channel first to channel last
    vgg2_fp = np.transpose(vgg2_fp, axes=[0, 2, 3, 1])

    import cv2

    ds = (
        tf.data.Dataset.from_tensor_slices(vgg2_fp)
        .map(preprocess_input, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
    )

    input_shape = (112, 112, 3)
    n_classes = 10575
    embedding_dimension = 512
    weight_decay = 5e-4

    model = create_model(
        input_shape=input_shape,
        n_classes=n_classes,
        embedding_dimension=embedding_dimension,
        weights_decay=weight_decay,
        use_pretrain=False,
    )
    model.load_weights(model_path)
    model = tf.keras.models.Model(
        inputs=model.inputs, outputs=model.get_layer("embedding").output
    )

    embeddings = model.predict(ds)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    dists = np.array(
        [np.linalg.norm(emb1 - emb2) for emb1, emb2 in chunked(embeddings, 2)]
    )

    for (im1, im2), is_same, dist in zip(chunked(vgg2_fp, 2), vgg2_fp_is_same, dists):
        window_name = f"{is_same} {dist:.5f}"
        cv2.imshow(window_name, cv2.hconcat([im1, im2]))
        key = cv2.waitKey()
        if key == ord("q"):
            return
        cv2.destroyWindow(window_name)

    plt.hist(
        [dists[vgg2_fp_is_same == 1], dists[vgg2_fp_is_same == 0]],
        bins=100,
        label=["same", "diff"],
    )
    plt.legend()
    plt.show()


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset-dir", "-d", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(args.model_path, args.dataset_dir)
