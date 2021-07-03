import unittest
import torch
from torch_impl.MultipleNegativeRankingLoss import MultipleNegativesRankingLoss
from trainer.loss.custom import multiple_negatives_ranking_loss
from jax import value_and_grad
from jax import random
import jax.numpy as jnp
import numpy as onp
import optax
from jax import nn as jax_nn
from torch import nn as torch_nn
from trainer.loss.custom import jax_cross_entropy_loss
from jax.config import config

config.update("jax_enable_x64", True)

class LossTest(unittest.TestCase):
    def test_jax_cross_entropy_loss(self):
        key = random.PRNGKey(0)
        key, a_key, b_key = random.split(key, 3)

        sample_count = 200
        label_count = 400
        scores = random.normal(a_key, (sample_count, label_count))
        torch_scores = torch.tensor(onp.asarray(scores), requires_grad=True)

        labels = random.randint(b_key, (sample_count, ), minval=0, maxval=label_count - 1)
        torch_labels = torch.tensor(onp.asarray(labels), dtype=torch.long)
        jax_labels = jax_nn.one_hot(labels, num_classes=label_count)

        jax_cross_entropy = jnp.mean(optax.softmax_cross_entropy(scores, jax_labels))
        jax_padded_cross_entropy = jax_cross_entropy_loss(scores, labels)
        torch_cross_entropy = torch_nn.CrossEntropyLoss()
        torch_cross_entropy = torch_cross_entropy.forward(torch_scores, torch_labels)

        assert onp.all(onp.abs(torch_cross_entropy.item() - jax_cross_entropy) < 0.001)
        assert onp.all(onp.abs(torch_cross_entropy.item() - jax_padded_cross_entropy) < 0.001)


    def test_multiple_negatives_ranking_loss(self):
        """Tests the correct computation of multiple_negatives_ranking_loss"""
        key = random.PRNGKey(0)
        key, a_key, b_key = random.split(key, 3)
        a = random.normal(a_key, (20, 200))
        b = random.normal(b_key, (20, 200))

        a_torch = torch.tensor(onp.asarray(a), requires_grad=True)
        b_torch = torch.tensor(onp.asarray(b), requires_grad=True)

        torch_loss = MultipleNegativesRankingLoss()
        torch_loss = torch_loss.forward(a_torch, b_torch, None)
        torch_loss.backward()
        torch_grad = a_torch.grad.numpy()

        jax_loss, jax_grad = value_and_grad(multiple_negatives_ranking_loss)(a, b)
        assert abs(torch_loss.item() - jax_loss) <= 0.001, "loss : {} vs {}".format(jax_loss, torch_loss.item())

        assert onp.all(onp.abs(torch_grad - jax_grad) < 0.001)

    def test_multiple_negatives_ranking_loss_triple(self):
        """Tests the correct computation of multiple_negatives_ranking_loss, with hard negatives"""
        key = random.PRNGKey(0)
        key, a_key, b_key, c_key = random.split(key, 4)
        a = random.normal(a_key, (20, 200))
        b = random.normal(b_key, (20, 200))
        c = random.normal(c_key, (20, 200))

        a_torch = torch.tensor(onp.asarray(a), requires_grad=True)
        b_torch = torch.tensor(onp.asarray(b), requires_grad=True)
        c_torch = torch.tensor(onp.asarray(c), requires_grad=True)

        comp_torch = torch.cat([b_torch, c_torch])

        torch_loss = MultipleNegativesRankingLoss()
        torch_loss = torch_loss.forward(a_torch, comp_torch, None)
        torch_loss.backward()
        torch_grad = a_torch.grad.numpy()

        jax_loss, jax_grad = value_and_grad(multiple_negatives_ranking_loss)(a, jnp.concatenate([b, c], axis=0))
        assert abs(torch_loss.item() - jax_loss) <= 0.001, "loss : {} vs {}".format(jax_loss, torch_loss.item())

        assert onp.all(onp.abs(torch_grad - jax_grad) < 0.001)