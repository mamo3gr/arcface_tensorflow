import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal

from arcface import ArcfaceLayer


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
