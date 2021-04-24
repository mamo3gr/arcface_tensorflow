import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal

from arcface import ArcfaceLayer, ArcFaceLoss


class TestArcfaceLayer:
    n_classes = 10
    seed = 42
    feature_dimension = 256
    batch_size = 8

    @pytest.fixture
    def layer(self):
        return ArcfaceLayer(n_classes=self.n_classes, seed=self.seed)

    @pytest.fixture
    def tensor(self):
        return tf.random.normal([self.batch_size, self.feature_dimension])

    def test_build(self, layer, tensor):
        # able to be tested at the same time as *call*
        pass

    def test_call(self, layer, tensor):
        out_actual = layer.__call__(inputs=tensor)

        # check created weight
        weights = layer.get_weights()
        assert len(weights) == 1
        w = weights[0]
        assert w.shape == (self.feature_dimension, self.n_classes)

        # check outcome
        out_expect = np.matmul(
            tensor.numpy() / np.linalg.norm(tensor.numpy(), axis=1, keepdims=True),
            w / np.linalg.norm(w, axis=0, keepdims=True),
        )
        assert_almost_equal(out_actual, out_expect)

    def test_get_config(self, layer):
        config = layer.get_config()
        assert config.get("n_classes") == self.n_classes
        assert config.get("seed") == self.seed


class TestArcFaceLoss:
    batch_size = 16
    n_classes = 32
    margin = 0.5
    scale = 60
    loss_func = tf.keras.losses.CategoricalCrossentropy()

    @pytest.fixture
    def loss(self):
        return ArcFaceLoss(
            loss_func=self.loss_func, margin=self.margin, scale=self.scale
        )

    def test_call(self, loss):
        y_true = self._random_onehot(self.batch_size, self.n_classes).astype(np.int)
        y_pred = self._random_cosine(self.batch_size, self.n_classes).astype(np.float32)

        loss_actual = loss.call(y_true, y_pred)

        cos_t_plus_m = np.cos(
            np.arccos(y_pred) + np.where(y_true > 0, np.cos(self.margin), 0)
        )
        softmax = self._softmax(cos_t_plus_m * self.scale, axis=1)
        loss_expect = self.loss_func(y_true, softmax.astype(np.float32))

        assert_almost_equal(loss_actual, loss_expect, decimal=3)

    @staticmethod
    def _random_onehot(n_samples: int, n_classes: int):
        return np.eye(n_classes)[np.random.choice(n_classes, n_samples)]

    @staticmethod
    def _random_cosine(n_samples: int, n_classes: int):
        """
        generate [-1, 1) randomly
        """
        a = -1
        b = 1
        return (b - a) * np.random.rand(n_samples, n_classes) + a

    @staticmethod
    def _softmax(x, axis=None):
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
