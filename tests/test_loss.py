import unittest
import torch
from torch_impl.MultipleNegativeRankingLoss import MultipleNegativesRankingLoss
from trainer.loss.custom import multiple_negatives_ranking_loss
from jax import value_and_grad
from jax import random
import jax.numpy as jnp
import numpy as onp

class LossTest(unittest.TestCase):
    def test_multiple_negatives_ranking_loss(self):
        """Tests the correct computation of multiple_negatives_ranking_loss"""
        key = random.PRNGKey(0)
        key, a_key, b_key = random.split(key, 3)
        a = random.normal(a_key, (20, 200))
        b = random.normal(b_key, (20, 200))

        a_torch = torch.tensor(onp.asarray(a), requires_grad=True)
        b_torch = torch.tensor(onp.asarray(b), requires_grad=True)

        jax_input = jnp.stack([a, b], axis=1)

        torch_loss = MultipleNegativesRankingLoss()
        torch_loss = torch_loss.forward(a_torch, b_torch, None)
        torch_loss.backward()
        torch_grad = torch.stack([a_torch.grad, b_torch.grad], dim=1).numpy()

        jax_loss, jax_grad = value_and_grad(multiple_negatives_ranking_loss)(jax_input)
        assert abs(torch_loss.item() - jax_loss) <= 0.05

        assert onp.all(onp.abs(torch_grad - jax_grad) < 0.01)