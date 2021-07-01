from typing import Callable
from flax import struct
from flax.training import train_state
from flax import jax_utils, traverse_util
import jax
import jax.numpy as jnp
import optax
from functools import partial
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    model_id: str
    max_epochs: int
    seed: int
    lr: float
    init_lr: float
    warmup_steps: int
    weight_decay: float


def scheduler_fn(lr, init_lr, warmup_steps, num_train_steps):
    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(init_value=init_lr, end_value=lr, transition_steps=warmup_steps)
    decay_fn = optax.linear_schedule(init_value=lr, end_value=1e-7, transition_steps=decay_steps)
    lr = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return lr


def build_tx(lr, init_lr, warmup_steps, num_train_steps, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.flatten_dict(params)
        mask = {k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale")) for k, v in params.items()}
        return traverse_util.unflatten_dict(mask)
    lr = scheduler_fn(lr, init_lr, warmup_steps, num_train_steps)
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask)
    return tx, lr


class TrainState(train_state.TrainState):
    loss_fn: Callable = struct.field(pytree_node=False)
    scheduler_fn: Callable = struct.field(pytree_node=False)


def multiple_negative_ranking_loss(embedding1, embedding2):
    def _cross_entropy(logits):
        bsz = logits.shape[-1]
        labels = (jnp.arange(bsz)[..., None] == jnp.arange(bsz)[None]).astype("f4")
        logits = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(labels * logits, axis=-1)
        return loss

    batch_similarity = jnp.dot(embedding1, jnp.transpose(embedding2))
    loss = _cross_entropy(batch_similarity)
    return jnp.mean(loss)


@jax.jit
def train_step(state, model_input1, model_input2, drp_rng):
    train = True
    new_drp_rng, drp_rng = jax.random.split(drp_rng, 2)

    @partial(jax.pmap, axis_name="batch")
    def forward(params, model_input1, model_input2, drp_rng):
        def _forward(model_input):
            embedding = state.apply_fn(**model_input, params=params, train=train, dropout_rng=drp_rng)
            embedding = jnp.mean(embedding, axis=1)
            embedding = embedding / jnp.maximum(jnp.sum(jnp.square(embedding), axis=-1), 1e-12)
            # gather all the embeddings on same device for getting benefit of big batch
            embedding = jax.lax.all_gather(embedding, axis_name="batch")
            return embedding

        embedding1, embedding2 = _forward(model_input1), _forward(model_input2)
        return embedding1, embedding2

    def loss_fn(params):
        embedding1, embedding2 = forward(params, model_input1, model_input2, drp_rng)
        embedding1 = jax_utils.unreplicate(embedding1)
        embedding2 = jax_utils.unreplicate(embedding2)
        return state.loss_fn(embedding1, embedding2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"tr_loss": loss, "lr": state.scheduler_fn(jax_utils.unreplicate(state.step))}
    return state, metrics, new_drp_rng


def get_batched_dataset(dataset):
    # return sharded dataset with generator
    return


if __name__ == '__main__':
    from transformers import AutoModel
    from datasets import Dataset

    args = TrainingArgs()
    model = AutoModel.from_pretrained(args.model_id)

    num_train_steps: int
    tr_dataset: Dataset

    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": num_train_steps,
        "weight_decay": args.weight_decay,
    }
    tx, lr = build_tx(**tx_args)

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        loss_fn=multiple_negative_ranking_loss,
        scheduler_fn=lr,
    )

    rng = jax.random.PRNGKey(args.seed)
    drp_rng, rng = jax.random.split(rng)
    for epoch in range(args.max_epochs):
        for batch in get_batched_dataset(tr_dataset):
            model_input1, model_input2 = batch
            state, metrics, drp_rng = train_step(state, model_input1, model_input2, drp_rng)
