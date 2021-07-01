import jax
from jax import numpy as jnp


@jax.jit
def cos_sim(a, b):
    a_norm = jnp.linalg.norm(a, ord=2)
    b_norm = jnp.linalg.norm(b, ord=2)
    return jnp.dot(a_norm, b_norm.T)
