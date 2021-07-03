import jax
from jax import numpy as jnp
import optax


@jax.jit
def jax_cross_entropy_loss(scores, labels):
    """
    :param scores:
    :param labels:
    :return:
    """
    class_count = scores.shape[1]
    assert scores.shape[0] == len(labels)
    one_hot = jax.nn.one_hot(labels, num_classes=class_count)
    soft_max = optax.softmax_cross_entropy(scores, one_hot)
    return jnp.mean(soft_max)
