from dataclasses import dataclass, field, asdict, replace
from functools import partial
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
from flax.training.common_utils import shard
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import wandb
import json
import os

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, FlaxAutoModel


@dataclass
class TrainingArgs:
    model_id: str = "bert-base-uncased"
    max_epochs: int = 2
    batch_size_per_device: int = 32 # it must be multiple of 8 (when on TPUs)
    seed: int = 42
    lr: float = 2e-5
    init_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 1e-3

    input1_maxlen: int = 128
    input2_maxlen: int = 128
    
    logging_steps: int = 20
    save_dir: str = "checkpoints"

    tr_data_files: List[str] = field(
        default_factory=lambda: [
            "data/dummy.jsonl",
        ]
    )
        
    val_data_files: List[str] = field(
        default_factory=lambda: [
            "data/dummy.jsonl",
        ]
    )

    def __post_init__(self):
        self.batch_size = self.batch_size_per_device * jax.device_count()


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

    metrics = {"tr_loss": loss, "lr": state.scheduler_fn(state.step)}
    return state, metrics, new_drp_rng


@partial(jax.pmap, axis_name="batch")
def val_step(state, model_inputs1, model_inputs2):
    train = False

    def _forward(model_input):
        attention_mask = model_input["attention_mask"][..., None]
        embedding = state.apply_fn(**model_input, params=state.params, train=train)[0]
        attention_mask = jnp.broadcast_to(attention_mask, jnp.shape(embedding))

        embedding = embedding * attention_mask
        embedding = jnp.mean(embedding, axis=1)

        modulus = jnp.sum(jnp.square(embedding), axis=-1, keepdims=True)
        embedding = embedding / jnp.maximum(modulus, 1e-12)

        # gather all the embeddings on same device for calculation loss over global batch
        embedding = jax.lax.all_gather(embedding, axis_name="batch")
        embedding = jnp.reshape(embedding, (-1, embedding.shape[-1]))

        return embedding

    embedding1, embedding2 = _forward(model_inputs1), _forward(model_inputs2)
    loss = state.loss_fn(embedding1, embedding2)
    return jnp.mean(loss)


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


def save_checkpoint(save_dir, state, save_fn=None, training_args=None):
    print(f"saving checkpoint in {save_dir}", end=" ... ")

    os.makedirs(save_dir, exist_ok=True)
    state = jax_utils.unreplicate(state)

    if save_fn is not None:
        # saving model in HF fashion
        save_fn(save_dir, params=state.params)
    else:
        path = os.path.join(save_dir, "flax_model.msgpack")
        with open(path, "wb") as f:
            f.write(to_bytes(state.params))

    # this will save optimizer states
    path = os.path.join(save_dir, "opt_state.msgpack")
    with open(path, "wb") as f:
        f.write(to_bytes(state.opt_state))

    if training_args is not None:
        path = os.path.join(save_dir, "training_args.json")
        with open(path, "w") as f:
            json.dump(asdict(training_args), f)

    print("done!!")


def prepare_dataset(args):
    tr_dataset = load_dataset("json", data_files=args.tr_data_files, split="train")
    val_dataset = load_dataset("json", data_files=args.val_data_files, split="train")

    # ensures similar processing to all splits at once
    dataset = DatasetDict(train=tr_dataset, validation=val_dataset)

    columns_to_remove = ['repo', 'path', 'func_name', 'original_string', 'sha', 'url', 'partition']
    dataset = dataset.remove_columns(columns_to_remove)

    # drop extra batch from the end
    for split in dataset:
        num_samples = len(dataset[split]) - len(dataset[split]) % args.batch_size
        dataset[split] = dataset[split].shuffle(seed=args.seed).select(range(num_samples))

    print(dataset)
    tr_dataset, val_dataset = dataset["train"], dataset["validation"]
    return tr_dataset, val_dataset

    
def main(args, logger):
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = FlaxAutoModel.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    data_collator = DataCollator(
        tokenizer=tokenizer,
        input1_maxlen=args.input1_maxlen,
        input2_maxlen=args.input2_maxlen,
    )

    tr_dataset, val_dataset = prepare_dataset(args)

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
        total = len(tr_dataset) // args.batch_size
        batch_iterator = get_batched_dataset(tr_dataset, args.batch_size, seed=epoch)
        for i, batch in tqdm(enumerate(batch_iterator), desc=f"Running epoch-{epoch}", total=total):
            model_input1, model_input2 = data_collator(batch)
            state, metrics, drp_rng = train_step(state, model_input1, model_input2, drp_rng)

            if (i + 1) % args.logging_steps == 0:
                tr_loss = jax_utils.unreplicate(metrics["tr_loss"]).item()
                tqdm.write(str(dict(tr_loss=tr_loss, step=i+1)))
                logger.log({
                    "tr_loss": tr_loss,
                    "step": i + 1,
                }, commit=True)

        # evaluation
        val_loss  = jnp.array(0.)
        total = len(val_dataset) // args.batch_size
        val_batch_iterator = get_batched_dataset(val_dataset, args.batch_size, seed=None)
        for j, batch in tqdm(enumerate(val_batch_iterator), desc=f"evaluating after epoch-{epoch}", total=total):
            model_input1, model_input2 = data_collator(batch)
            val_step_loss = val_step(state, model_input1, model_input2)
            val_loss += jax_utils.unreplicate(val_step_loss)

        val_loss = val_loss.item() / (j + 1)
        print(f"val_loss: {val_loss}")
        logger.log({"val_loss": val_loss}, commit=True)
        
        save_dir = args.save_dir + f"-epoch-{epoch}"
        save_checkpoint(save_dir, state, save_fn=model.save_pretrained, training_args=args)


if __name__ == '__main__':

    args = TrainingArgs()
    logger = wandb.init(project="code-search-net", config=asdict(args))
    logging_dict = dict(logger.config); logging_dict["save_dir"] += f"-{logger.id}"
    args = replace(args, **logging_dict)

    print(args)
    main(args, logger)
