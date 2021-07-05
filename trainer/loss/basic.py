import jax
from jax import numpy as jnp
import optax


@jax.jit
def jax_cross_entropy_loss(scores, labels):
    class_count = scores.shape[1]
    assert scores.shape[0] == len(labels)
    one_hot = jax.nn.one_hot(labels, num_classes=class_count)
    soft_max = optax.softmax_cross_entropy(scores, one_hot)
    return jnp.mean(soft_max)




def cross_entropy(logits, axis):
    """
    https://github.com/huggingface/transformers/blob/4605b2b8ec5512a5ea125773bcaa4b0014b32d50/examples/research_projects/jax-projects/hybrid_clip/run_hybrid_clip.py#L430
    """
    logprobs = jax.nn.log_softmax(logits, axis=axis)
    nll = jnp.diag(logprobs)
    ce = -jnp.mean(nll)
    return ce
