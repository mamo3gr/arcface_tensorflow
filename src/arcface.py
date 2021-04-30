from math import cos, sin
from typing import Optional

import tensorflow as tf


class Angle(tf.keras.layers.Layer):
    """
    Angles to a kind of centre for each class
    (More precisely, not angle but cosine of the angle).
    """

    def __init__(
        self,
        n_classes: int,
        initializer: Optional[tf.keras.initializers.Initializer] = None,
        regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super(Angle, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.n_classes),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
        )
        super(Angle, self).build(input_shape)

    def call(self, inputs, **kwargs):
        embeddings = inputs
        normalized_embeddings = tf.math.l2_normalize(
            embeddings, axis=1, name="normalized_embeddings"
        )
        normalized_weight = tf.math.l2_normalize(
            self.weight, axis=0, name="normalized_weight"
        )
        cos_t = tf.matmul(normalized_embeddings, normalized_weight, name="cos_t")
        return cos_t

    def get_config(self):
        config = super(Angle, self).get_config()
        config.update(
            {
                "n_classes": self.n_classes,
                "initializer": self.initializer,
                "regularizer": self.regularizer,
            }
        )
        return config


class AdditiveAngularMarginLoss(tf.keras.losses.Loss):
    def __init__(self, loss_func, margin: float, scale: float, **kwargs):
        self.loss_func = loss_func
        self.margin = margin
        self.scale = scale
        self.cos_m = tf.constant(cos(self.margin))
        self.sin_m = tf.constant(sin(self.margin))
        super(AdditiveAngularMarginLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        cos_t = y_pred
        sin_t = tf.sqrt(1 - cos_t ** 2)

        # cos(t + m) = cos(t)cos(m) - sin(t)sin(m)
        cos_t_plus_m = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m)
        cos_t_plus_m = tf.where(cos_t_plus_m > 0, cos_t_plus_m, cos_t)

        logits = tf.where(y_true > 0, cos_t_plus_m, cos_t)
        logits = tf.multiply(logits, self.scale)
        softmax = tf.nn.softmax(logits)
        return self.loss_func(y_true, softmax)
