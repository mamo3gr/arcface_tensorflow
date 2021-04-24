import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal


class ArcfaceLayer(tf.keras.layers.Layer):
    def __init__(self, n_classes: int, seed=42, **kwargs):
        super(ArcfaceLayer, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.seed = seed

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.n_classes),
            initializer=TruncatedNormal(seed=self.seed),
            trainable=True,
        )
        super(ArcfaceLayer, self).build(input_shape)

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
        config = super(ArcfaceLayer, self).get_config()
        config.update({"n_classes": self.n_classes, "seed": self.seed})
        return config
