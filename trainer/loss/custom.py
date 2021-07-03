import jax
from jax import numpy as jnp
from .basic import padded_cross_entropy_loss
from ..utils.ops import cos_sim


@jax.jit
def multiple_negatives_ranking_loss_single_input(embeddings: jnp.DeviceArray, scale: float = 20.0,
                                                 similarity_fct=cos_sim):
    embeddings_a = embeddings[:, 0, :]
    # positive and hard negatives (if any, flattened and treated as additional samples).
    embeddings_b = jnp.reshape(embeddings[:, 1:, :], (-1, embeddings.shape[-1]))

    assert (len(embeddings_a) <= len(embeddings_b))
    scores = similarity_fct(embeddings_a, embeddings_b) * scale
    assert scores.shape == (len(embeddings_a), len(embeddings_b))

    labels = jnp.arange(len(embeddings_a), dtype=jnp.int64)
    return padded_cross_entropy_loss(scores, labels)


@jax.jit
def multiple_negatives_ranking_loss(embeddings_a: jnp.DeviceArray, embeddings_b: jnp.DeviceArray,
                                    scale: float = 20.0, similarity_fct=cos_sim):
    assert (len(embeddings_a) <= len(embeddings_b))
    scores = similarity_fct(embeddings_a, embeddings_b) * scale
    assert scores.shape == (len(embeddings_a), len(embeddings_b))

    labels = jnp.arange(len(embeddings_a), dtype=jnp.int64)
    return padded_cross_entropy_loss(scores, labels)
