from logging import getLogger

import tensorflow as tf

logger = getLogger(__name__)


def set_gpu_memory_growth():
    """
    Reference:
      Use a GPU | TensorFlow Core
      https://www.tensorflow.org/guide/gpu
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logger.info(
                f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}"
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.warning(e)


def get_strategy():
    """
    References:
      Tensorflow.Kerasモデルで TPU/GPU/CPU を自動的に切り替える - Qiita
      https://qiita.com/aizakku_nidaa/items/686d0544d0201a4e3610
    """
    tpu = None
    gpus = []

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError:
        gpus = tf.config.experimental.list_logical_devices("GPU")

    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
        # Going back and forth between TPU and host is expensive.
        # Better to run 128 batches on the TPU before reporting back.
        strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
        logger.info("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        logger.info("Running on multiple GPUs ", [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = (
            tf.distribute.get_strategy()
        )  # default strategy that works on CPU and single GPU
        logger.info("Running on single GPU ", gpus[0].name)
    else:
        strategy = (
            tf.distribute.get_strategy()
        )  # default strategy that works on CPU and single GPU
        logger.info("Running on CPU")

    logger.info("Number of accelerators: ", strategy.num_replicas_in_sync)

    return strategy


if __name__ == "__main__":
    set_gpu_memory_growth()
