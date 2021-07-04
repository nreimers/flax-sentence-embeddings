import jax
from jax import random, numpy as jnp
from flax import linen as nn
from trainer.loss.custom import multiple_negatives_ranking_loss
from jax.config import config

# Dummy version
batch_size = 20
embedding_size = 250


def demo_train_step(model, params, input):
    # We can integrate with existing scripts. this is for demo purpose.

    def loss(params):
        preds = model.apply(params, input)
        preds = jnp.reshape(preds, (preds.shape[0], -1, embedding_size))
        return multiple_negatives_ranking_loss(preds)

    loss, grad = jax.value_and_grad(loss)(params)
    return loss, grad


def main():
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    dummy_model = nn.Dense(features=3 * embedding_size)
    dummy_input = random.normal(key1, (batch_size, 200))
    params = dummy_model.init(key2, dummy_input)

    value, grad = demo_train_step(dummy_model, params, dummy_input)
    print("Value : ", value)
    print("Grad : ", grad)


if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    main()
