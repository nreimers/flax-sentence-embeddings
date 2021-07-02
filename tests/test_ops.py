from trainer.utils.ops import cos_sim, mean_pooling
import jax.numpy as jnp
import numpy as onp

def test_cos():
    a = jnp.array([[0, 1, 1, 0]])
    b = jnp.array([[1, 1, 1, 2]])

    assert onp.isclose(onp.array(cos_sim(a, a)), 1)
    assert onp.array(cos_sim(a, b)) < 1


