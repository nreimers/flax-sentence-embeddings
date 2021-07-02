import jax


@jax.jit
def cos_sim(a, b):
    return (a @ b.T).T
