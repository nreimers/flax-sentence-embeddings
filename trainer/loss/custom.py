import jax
from jax import numpy as jnp
from basic import cross_entropy_loss
from ..utils.ops import cos_sim


@jax.jit
def multiple_negatives_ranking_loss(embeddings: jnp.DeviceArray, scale: float = 20.0,
                                    similarity_fct=cos_sim):
    embeddings_a = embeddings[0]
    embeddings_b = embeddings[1]

    # TODO : Support size 3 (additional negatives)

    # No need to think about batch_sizes as long as it is at first.
    scores = jax.vmap(similarity_fct)(embeddings_a, embeddings_b) * scale
    labels = jnp.arange(len(scores), dtype=jnp.uint64)
    return cross_entropy_loss(scores, labels)
