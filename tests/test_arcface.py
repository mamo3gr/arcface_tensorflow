import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal

from arcface import AdditiveAngularMarginLoss, Angle


class TestAngleLayer:
    n_classes = 10
    seed = 42
    feature_dimension = 256
    batch_size = 8
    weights_decay = 5e-4
    regularizer = tf.keras.regularizers.l2(weights_decay)

    @pytest.fixture
    def layer(self):
        return Angle(
            n_classes=self.n_classes, regularizer=self.regularizer, seed=self.seed
        )

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
        assert config.get("regularizer") == self.regularizer
        assert config.get("seed") == self.seed


class TestArcFaceLoss:
    margin = 0.2 * np.pi
    scale = 5
    loss_func = tf.keras.losses.CategoricalCrossentropy()

    @pytest.fixture
    def loss(self):
        return AdditiveAngularMarginLoss(
            loss_func=self.loss_func, margin=self.margin, scale=self.scale
        )

    def test_call(self, loss):
        # fmt: off
        y_true = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.int)
        angles = np.array([
            [0.25 * np.pi, 0.20 * np.pi],
            [0.45 * np.pi, 0.10 * np.pi],  # 0.45pi + 0.20pi (margin) > 0.5pi = pi/2
        ], dtype=np.float32)
        # fmt: on
        y_pred = np.cos(angles)

        # add margin to theta (angle) if corresponding class is true positive
        cos_t_plus_m = np.cos(angles + np.where((y_true == 1), self.margin, 0))
        # if cos(t + m) is negative, use cos(t)
        cos_t_plus_m = np.where(cos_t_plus_m < 0, np.cos(angles), cos_t_plus_m)
        softmax = self._softmax(cos_t_plus_m * self.scale, axis=1)
        loss_expect = self.loss_func(y_true, softmax.astype(np.float32))

        loss_actual = loss.call(y_true, y_pred)
        assert_almost_equal(loss_actual, loss_expect)

    @staticmethod
    def _softmax(x, axis=None):
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
