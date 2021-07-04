import trainer.utils.ops as jax_util
import numpy as onp
from torch_impl import util as torch_util
import unittest


class UtilTest(unittest.TestCase):
    def test_cos_sim(self):
        """Tests the correct computation of utils.ops.cos_sim"""
        a = onp.random.randn(50, 100)
        b = onp.random.randn(50, 100)

        pytorch_cos_scores = torch_util.cos_sim(a, b).numpy()
        jax_cos_scores = onp.asarray(jax_util.cos_sim(a, b))

        assert pytorch_cos_scores.shape == jax_cos_scores.shape
        for i in range(len(jax_cos_scores)):
            for j in range(len(jax_cos_scores[0])):
                assert abs(pytorch_cos_scores[i][j] - jax_cos_scores[i][j]) < 0.001, "Output : torch - {}, jax - {}" \
                    .format(pytorch_cos_scores[i], jax_cos_scores[i])
