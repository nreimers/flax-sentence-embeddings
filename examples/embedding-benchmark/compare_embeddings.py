from transformers import AutoTokenizer, AutoModel, FlaxAutoModel
import sys
import torch
import jax
from jax import numpy as jnp
#import numpy as np
from sentence_transformers import util
import numpy as np
import sys
sys.path.append("../..")

from trainer.utils.ops import normalize_L2, mean_pooling

def p_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FlaxSEModel:
    def __init__(self, model_name):
        self.model = jax.jit(FlaxAutoModel.from_pretrained(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = 128

    def encode(self, sentences, batch_size=32, convert_to_tensor=False, **kwargs):
        all_emb = []

        for start_idx in range(0, len(sentences), batch_size):
            emb = self._encode_batch(sentences[start_idx:start_idx + batch_size])
            all_emb.append(emb)

        all_emb = np.concatenate(all_emb, axis=0)

        if convert_to_tensor:
            all_emb = torch.tensor(all_emb)

        return all_emb

    def _encode_batch(self, sentences):
        encoded_input = self.tokenizer(sentences, padding='max_length', truncation='longest_first', return_tensors="jax", max_length=self.max_seq_length)
        model_output = self.model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = normalize_L2(embedding)
        return embedding

model_name = sys.argv[1]

flax_se = FlaxSEModel(model_name)
embedding = flax_se.encode(["Hello word!"])
print("Flax", embedding)

p_model = AutoModel.from_pretrained(model_name, from_flax=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
p_model.eval()
with torch.no_grad():
    encoded_input = tokenizer(["Hello word!"], return_tensors="pt")
    p_out = p_model(**encoded_input)
    p_emb = p_mean_pooling(p_out, encoded_input['attention_mask'])
print("Pytorch", p_emb)

print("Cos", util.cos_sim(np.array(embedding), p_emb))