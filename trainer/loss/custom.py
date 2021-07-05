import jax
from jax import numpy as jnp
from .basic import jax_cross_entropy_loss, cross_entropy
from ..utils.ops import cos_sim


@jax.jit
def multiple_negatives_ranking_loss(embeddings_a: jnp.DeviceArray, embeddings_b: jnp.DeviceArray,
                                    scale: float = 20.0, similarity_fct=cos_sim):
    """

    :param embeddings_a:
    :param embeddings_b: if passing additional hard negatives, use jnp.concatenate([positives, negatives], axis=0) as input.
    :param scale:
    :param similarity_fct:
    :return:
    """
    assert (len(embeddings_a) <= len(embeddings_b))
    scores = similarity_fct(embeddings_a, embeddings_b) * scale
    assert scores.shape == (len(embeddings_a), len(embeddings_b))

    labels = jnp.arange(len(scores), dtype=jnp.int32)
    return jax_cross_entropy_loss(scores, labels)

    """
    loss = (cross_entropy(scores, axis=0) + cross_entropy(scores, axis=1)) / 2
    return loss
    """