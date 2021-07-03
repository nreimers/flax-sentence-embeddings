import jax
from jax import numpy as jnp


@jax.jit
def cos_sim(a, b):
    a = a / jnp.maximum(jnp.linalg.norm(a, ord=2, axis=1, keepdims=True), 1e-12)
    b = b / jnp.maximum(jnp.linalg.norm(b, ord=2, axis=1, keepdims=True), 1e-12)
    return a @ b.T


@jax.jit
def mean_pooling(model_output, attention_mask):
    """
    The function applies mean poling to a set of token embeddings

    :param model_output: last_hidden_state of the flax Transformers
    :param attention_mask: the attention mask to zero out some vectors
    :return: the mean pooled embeddings
    """
    input_mask_expanded = jnp.repeat(jnp.expand_dims(attention_mask, -1), model_output.shape[2], axis=2)
    return jnp.sum(model_output * input_mask_expanded, 1) / jnp.clip(input_mask_expanded.sum(1), a_min=1e-9)
