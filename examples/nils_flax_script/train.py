
from jax.config import config

from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Callable, List, Union
import wandb
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from flax.training.common_utils import shard
from flax.serialization import to_bytes, from_bytes
from tqdm.auto import tqdm

from datetime import datetime

from transformers import AutoTokenizer, FlaxAutoModel
import gzip
import json
import logging
from MultiDatasetDataLoader import MultiDatasetDataLoader, InputExample

import argparse
import os
import sys

sys.path.append("../..")
from trainer.loss.custom import multiple_negatives_ranking_loss







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
        return state.loss_fn(embedding1, embedding2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, model_input1, model_input2, drp_rng)
    state = state.apply_gradients(grads=grads)

    metrics = {"train_loss": loss, "lr": state.scheduler_fn(state.step)}
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


def save_checkpoint(save_dir, state, tokenizer, save_fn=None, training_args=None):
    print(f"saving checkpoint in {save_dir}", end=" ... ")

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)

    state = jax_utils.unreplicate(state)

    if save_fn is not None:
        # saving model in HF fashion
        save_fn(save_dir, params=state.params)
    else:
        path = os.path.join(save_dir, "flax_model.msgpack")
        with open(path, "wb") as f:
            f.write(to_bytes(state.params))

    # this will save optimizer states
    """
    path = os.path.join(save_dir, "opt_state.msgpack")
    with open(path, "wb") as f:
        f.write(to_bytes(state.opt_state))
    """

    if training_args is not None:
        path = os.path.join(save_dir, "training_args.json")
        with open(path, "w") as f:
            json.dump(vars(training_args), f)

    print("done!!")



def data_collator(batch, tokenizer):
    texts1 = [e.texts[0] for e in batch]
    texts2 = [e.texts[1] for e in batch]

    model_input1 = tokenizer(texts1, return_tensors="jax", max_length=128, truncation=True, padding=True, pad_to_multiple_of=64)
    model_input2 = tokenizer(texts2, return_tensors="jax", max_length=128, truncation=True, padding=True, pad_to_multiple_of=64)
    model_input1, model_input2 = dict(model_input1), dict(model_input2)
    return shard(model_input1), shard(model_input2)


def main(args, train_dataloader):
    model = FlaxAutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": args.steps,
        "weight_decay": args.weight_decay,
    }
    tx, lr = build_tx(**tx_args)

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        loss_fn=multiple_negatives_ranking_loss,
        scheduler_fn=lr,
    )
    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(args.seed)
    drp_rng = jax.random.split(rng, jax.device_count())

    print("Train steps:", len(train_dataloader))
    global_step = 0
    for epoch in range(args.epochs):
        # training step
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Running epoch-{epoch}"):
            model_input1, model_input2 = data_collator(batch, tokenizer)
            state, metrics, drp_rng = train_step(state, model_input1, model_input2, drp_rng)
            global_step += 1

            if global_step % args.logging_steps == 0:
                train_loss = jax_utils.unreplicate(metrics["train_loss"]).item()
                data_log = {
                    "train_loss": train_loss,
                    "global_step": global_step,
                }
                tqdm.write(str(data_log))
                wandb.log(data_log)

            if global_step % args.save_steps == 0:
                save_dir = os.path.join(args.save_dir, f"steps-{global_step}")
                save_checkpoint(save_dir, state, tokenizer, save_fn=model.save_pretrained, training_args=args)

        # evaluation step
        # for batch in get_batched_dataset(val_dataset, args.batch_size, seed=None):
        #     model_input1, model_input2 = data_collator(batch)
        #     state, metric = val_step(state, model_input1, model_input2)

    save_dir = os.path.join(args.save_dir, "final")
    save_checkpoint(save_dir, state, tokenizer, save_fn=model.save_pretrained, training_args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size_pairs', type=int, default=256)
    parser.add_argument('--batch_size_triplets', type=int, default=256)
    parser.add_argument('--input1_maxlen', type=int, default=128)
    parser.add_argument('--input2_maxlen', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--random_batch_fraction', type=float, default=0.1)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_dir', default='output')
    parser.add_argument('--data', nargs='+', default=[])
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(args.save_dir)

    wandb.init(project="nils-test")
    wandb.config.update(args)

    datasets = []
    for filepath in args.data:
        filepath = filepath.strip()
        dataset = []

        with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                data = json.loads(line.strip())

                if not isinstance(data, dict):
                    data = {'guid': None, 'texts': data}

                dataset.append(InputExample(guid=data.get('guid', None), texts=data['texts']))
                if len(dataset) >= (args.steps * args.batch_size_pairs * 2):
                    break

        datasets.append(dataset)
        logging.info("{}: {}".format(filepath, len(dataset)))

    train_dataloader = MultiDatasetDataLoader(datasets,
                                              batch_size_pairs=args.batch_size_pairs,
                                              batch_size_triplets=args.batch_size_triplets,
                                              random_batch_fraction=args.random_batch_fraction,
                                              dataloader_len=args.steps)


    main(args, train_dataloader)