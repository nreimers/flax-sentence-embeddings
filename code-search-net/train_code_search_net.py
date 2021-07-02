from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class TrainingArgs:
    model_id: str = "bert-base-uncased"
    max_epochs: int = 2
    batch_size: int = 2
    seed: int = 42
    lr: float = 2e-5
    init_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 1e-3

    input1_maxlen: int = 128
    input2_maxlen: int = 128

    tr_data_files: List[str] = field(
        default_factory=lambda: [
            "data/dummy.jsonl",
        ]
    )


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
    return _cross_entropy(batch_similarity)


@partial(jax.pmap, axis_name="batch")
def train_step(state, model_input1, model_input2, drp_rng):
    train = True
    new_drp_rng, drp_rng = jax.random.split(drp_rng, 2)

    def loss_fn(params, model_input1, model_input2, drp_rng):
        def _forward(model_input):
            attention_mask = model_input["attention_mask"][..., None]
            embedding = state.apply_fn(**model_input, params=params, train=train, dropout_rng=drp_rng)[0]
            attention_mask = jnp.broadcast_to(attention_mask, jnp.shape(embedding))

            embedding = embedding * attention_mask
            embedding = jnp.mean(embedding, axis=1)

            modulus = jnp.sum(jnp.square(embedding), axis=-1, keepdims=True)
            embedding = embedding / jnp.maximum(modulus, 1e-12)

            # gather all the embeddings on same device for calculation loss over global batch
            embedding = jax.lax.all_gather(embedding, axis_name="batch")
            embedding = jnp.reshape(embedding, (-1, embedding.shape[-1]))

            return embedding

        embedding1, embedding2 = _forward(model_input1), _forward(model_input2)
        loss = state.loss_fn(embedding1, embedding2)
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, model_input1, model_input2, drp_rng)
    state = state.apply_gradients(grads=grads)

    step = jax.lax.pmean(state.step, axis_name="batch")
    metrics = {"tr_loss": loss, "lr": state.scheduler_fn(step)}

    return state, metrics, new_drp_rng


def get_batched_dataset(dataset, batch_size, seed=None):
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    for i in range(len(dataset) // batch_size):
        batch = dataset[i*batch_size: (i+1)*batch_size]
        yield dict(batch)


@dataclass
class DataCollator:
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
    input1_maxlen: int = 128
    input2_maxlen: int = 128

    def __call__(self, batch):
        # Currently only static padding; TODO: change below for adding dynamic padding support
        model_input1 = self.tokenizer(batch["docstring"], return_tensors="jax", max_length=self.input1_maxlen, truncation=True, padding="max_length")
        model_input2 = self.tokenizer(batch["code"], return_tensors="jax", max_length=self.input2_maxlen, truncation=True, padding="max_length")
        model_input1, model_input2 = dict(model_input1), dict(model_input2)
        return shard(model_input1), shard(model_input2)


def main(args):
    # code is generic to any other model as well
    model = FlaxBertModel.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    data_collator = DataCollator(
        tokenizer=tokenizer,
        input1_maxlen=args.input1_maxlen,
        input2_maxlen=args.input2_maxlen,
    )

    tr_dataset = load_dataset("json", data_files=args.tr_data_files, split="train")
    columns_to_remove = ['repo', 'path', 'func_name', 'original_string', 'sha', 'url', 'partition']
    tr_dataset = tr_dataset.remove_columns(columns_to_remove)
    # drop extra batch from the end
    num_tr_samples = len(tr_dataset) - len(tr_dataset) % args.batch_size
    tr_dataset = tr_dataset.shuffle(seed=args.seed).select(range(num_tr_samples))
    print(tr_dataset)

    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": (len(tr_dataset) // args.batch_size) * args.max_epochs,
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
    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(args.seed)
    drp_rng = jax.random.split(rng, jax.device_count())
    for epoch in range(args.max_epochs):
        # training step
        batch_iterator = get_batched_dataset(tr_dataset, args.batch_size, seed=epoch)
        for batch in tqdm(batch_iterator, desc=f"Running epoch-{epoch}"):
            model_input1, model_input2 = data_collator(batch)
            state, metrics, drp_rng = train_step(state, model_input1, model_input2, drp_rng)

        # evaluation step
        # for batch in get_batched_dataset(val_dataset, args.batch_size, seed=None):
        #     model_input1, model_input2 = data_collator(batch)
        #     state, metric = val_step(state, model_input1, model_input2)


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer, FlaxBertModel

    args = TrainingArgs()
    main(args)
