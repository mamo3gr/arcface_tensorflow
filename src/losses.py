import tensorflow as tf


class ClippedValueLoss(tf.keras.losses.Loss):
    def __init__(self, loss_func, x_min: float, x_max: float, **kwargs):
        self.loss_func = loss_func
        self.x_min = x_min
        self.x_max = x_max
        super(ClippedValueLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.x_min, self.x_max)
        return self.loss_func(y_true, y_pred)
