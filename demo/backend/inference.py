from sentence_transformers import SentenceTransformer
import pandas as pd
import jax.numpy as jnp

from typing import List
import config

# We download the models we will be using.
# If you do not want to use all, you can comment the unused ones.
distilroberta_model = SentenceTransformer(config.MODELS_ID['distilroberta'])
mpnet_model = SentenceTransformer(config.MODELS_ID['mpnet'])
minilm_l6_model = SentenceTransformer(config.MODELS_ID['minilm_l6'])

# Defining cosine similarity using flax.
def cos_sim(a, b):
    return jnp.matmul(a, jnp.transpose(b))/(jnp.linalg.norm(a)*jnp.linalg.norm(b))


# We get similarity between embeddings.
def text_similarity(anchor: str, inputs: List[str], model: str = 'distilroberta'):

    # Creating embeddings
    if model == 'distilroberta':
        anchor_emb = distilroberta_model.encode(anchor)[None, :]
        inputs_emb = distilroberta_model.encode([input for input in inputs])
    elif model == 'mpnet':
        anchor_emb = mpnet_model.encode(anchor)[None, :]
        inputs_emb = mpnet_model.encode([input for input in inputs])
    elif model == 'minilm_l6':
        anchor_emb = minilm_l6_model.encode(anchor)[None, :]
        inputs_emb = minilm_l6_model.encode([input for input in inputs])

    # Obtaining similarity
    similarity = list(jnp.squeeze(cos_sim(anchor_emb, inputs_emb)))

    # Returning a Pandas' dataframe
    d = {'inputs': [input for input in inputs],
         'score': [round(similarity[i],3) for i in range(len(similarity))]}
    df = pd.DataFrame(d, columns=['inputs', 'score'])

    return df.sort_values('score', ascending=False)
