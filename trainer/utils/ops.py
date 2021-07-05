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
    This function applies mean pooling to contextualized embeddings produced by the last layer of a Flax based HuggingFace Transformer model.

    :param model_output: model output from a model of type `FlaxPreTrainedModel`
    :param attention_mask: attention mask in the model input
    :return: mean pooled embeddings
    """
    embeddings = model_output[0]
    attention_mask_expanded = jnp.broadcast_to(jnp.expand_dims(attention_mask, -1), embeddings.shape)
    sum_embeddings = jnp.sum(embeddings * attention_mask_expanded, 1)
    sum_mask = jnp.clip(attention_mask_expanded.sum(1), a_min=1e-9)
    return sum_embeddings / sum_mask

@jax.jit
def max_pooling(model_output, attention_mask):
    """
    This function applies max pooling to contextualized embeddings produced by the last layer of a Flax based HuggingFace Transformer model.

    :param model_output: model output from a model of type `FlaxPreTrainedModel`
    :param attention_mask: attention mask in the model input
    :return: max pooled embeddings
    """
    embeddings = model_output[0]
    attention_mask_expanded = jnp.broadcast_to(jnp.expand_dims(attention_mask, -1), embeddings.shape)
    return jnp.max(embeddings * attention_mask_expanded, 1)

@jax.jit
def cls_pooling(model_output):
    """
    This function returns the [CLS] token embedding produced by the last layer of a Flax based HuggingFace Transformer model.

    :param model_output: model output from a model of type `FlaxPreTrainedModel`
    :param attention_mask: attention mask in the model input
    :return: [CLS] token embedding
    """
    return model_output[0][:, 0]
