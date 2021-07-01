import jax
from jax import numpy as jnp
import optax


@jax.jit
def cross_entropy_loss(scores, labels):
    return jnp.mean(optax.softmax_cross_entropy(scores, jax.nn.one_hot(labels, num_classes=len(labels))))
