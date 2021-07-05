import unittest
import numpy as onp
import jax.numpy as jnp
import trainer.utils.ops as jax_util
import torch
from torch_impl import util as torch_util


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

    def test_mean_pooling(self):
        """Tests the correct computation of mean_pooling"""
        batch_size = 3
        max_seq_len = 128
        embedding_size = 768
        
        model_outputs = (onp.random.randn(batch_size, max_seq_len, embedding_size), )
        attention_mask = onp.random.randint(2, size=(batch_size, max_seq_len))

        model_outputs_pt = torch.tensor(model_outputs)
        attention_mask_pt = torch.tensor(attention_mask)
        embeddings_pt = torch_util.mean_pooling(model_outputs_pt, attention_mask_pt)

        model_outputs_jax = jnp.asarray(model_outputs)
        attention_mask_jax = jnp.asarray(attention_mask)
        embeddings_jax = jax_util.mean_pooling(model_outputs_jax, attention_mask_jax)    
                    
        assert embeddings_pt.numpy().shape == onp.array(embeddings_jax).shape
        assert onp.all(onp.abs(embeddings_pt.numpy() - onp.array(embeddings_jax)) < 0.001)
    
    def test_max_pooling(self):
        """Tests the correct computation of max_pooling"""
        batch_size = 3
        max_seq_len = 128
        embedding_size = 768
        
        model_outputs = (onp.random.randn(batch_size, max_seq_len, embedding_size), )
        attention_mask = onp.random.randint(2, size=(batch_size, max_seq_len))

        model_outputs_pt = torch.tensor(model_outputs)
        attention_mask_pt = torch.tensor(attention_mask)
        embeddings_pt = torch_util.max_pooling(model_outputs_pt, attention_mask_pt)    

        model_outputs_jax = jnp.asarray(model_outputs)
        attention_mask_jax = jnp.asarray(attention_mask)
        embeddings_jax = jax_util.max_pooling(model_outputs_jax, attention_mask_jax)
                    
        assert embeddings_pt.numpy().shape == onp.array(embeddings_jax).shape
        assert onp.all(onp.abs(embeddings_pt.numpy() - onp.array(embeddings_jax)) < 0.001)

    def test_max_pooling(self):
        """Tests the correct computation of max_pooling"""
        batch_size = 3
        max_seq_len = 128
        embedding_size = 768
        
        model_outputs = (onp.random.randn(batch_size, max_seq_len, embedding_size), )
        attention_mask = onp.random.randint(2, size=(batch_size, max_seq_len))

        model_outputs_pt = torch.tensor(model_outputs)
        attention_mask_pt = torch.tensor(attention_mask)
        embeddings_pt = torch_util.max_pooling(model_outputs_pt, attention_mask_pt)    

        model_outputs_jax = jnp.asarray(model_outputs)
        attention_mask_jax = jnp.asarray(attention_mask)
        embeddings_jax = jax_util.max_pooling(model_outputs_jax, attention_mask_jax)
                    
        assert embeddings_pt.numpy().shape == onp.array(embeddings_jax).shape
        assert onp.all(onp.abs(embeddings_pt.numpy() - onp.array(embeddings_jax)) < 0.001)

    def test_cls_pooling(self):
        """Tests the correct computation of cls_pooling"""
        batch_size = 3
        max_seq_len = 128
        embedding_size = 768
        
        model_outputs = (onp.random.randn(batch_size, max_seq_len, embedding_size), )        

        model_outputs_pt = torch.tensor(model_outputs)        
        embeddings_pt = torch_util.cls_pooling(model_outputs_pt)    

        model_outputs_jax = jnp.asarray(model_outputs)        
        embeddings_jax = jax_util.cls_pooling(model_outputs_jax)
                    
        assert embeddings_pt.numpy().shape == onp.array(embeddings_jax).shape
        assert onp.all(onp.abs(embeddings_pt.numpy() - onp.array(embeddings_jax)) < 0.001)        