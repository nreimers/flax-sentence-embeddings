#from transformers import TextClassificationPipeline, AutoTokenizers
#from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
#import torch
import pandas as pd
import jax.numpy as jnp

from typing import List
import config


#code_classification_model = TextClassificationPipeline(
#  model=RobertaForSequenceClassification.from_pretrained(config.MODELS_ID['code_classification']),
#  tokenizer=RobertaTokenizer.from_pretrained(MODEL_ID))

#code_search_model = SentenceTransformer(config.MODELS_ID['code_search'])

distilroberta_model = SentenceTransformer(config.MODELS_ID['distilroberta'])
#mpnet_model = SentenceTransformer(config.MODELS_ID['mpnet'])




#def code_classification_inference(code: str):
#    return code_classification_model(code)


#def code_search_inference(query: str):
#    docstrings_emb = config.DATASETS['code_search']['docstrings_embedding']
#    query_emb = code_search_model.encode(query, convert_to_tensor=True)

#    hits = torch.nn.functional.cosine_similarity(query_emb[None, :], docstrings_emb, dim=1, eps=1e-8)
#    hit_id = torch.argmax(hits).item()

#    hit_code = config.DATASETS['code_search']['docstrings_codes_lists'][hit_id][1]
#
#    return hit_code


# Obtaining cosine similarity
def cos_sim(a,b):
    return jnp.matmul(a, jnp.transpose(b))/(jnp.linalg.norm(a)*jnp.linalg.norm(b))


def similarity_distilroberta(anchor: str, inputs: List[str]):

    # Creating embeddings
    anchor_emb = distilroberta_model.encode(anchor)[None, :]
    inputs_emb = distilroberta_model.encode([input for input in inputs])

    # Obtaining similarity
    similarity = list(jnp.squeeze(cos_sim(anchor_emb, inputs_emb)))

    # Returning a Pandas' dataframe
    d = {'inputs': [input for input in inputs],
         'score': [round(similarity[i],3) for i in range(len(similarity))]}
    df = pd.DataFrame(d, columns=['inputs', 'score'])

    return df.sort_values('score', ascending=False)


def similarity_mpnet(anchor: str, inputs: List[str]):

    # Creating embeddings
    anchor_emb = mpnet_model.encode(anchor)[None, :]
    inputs_emb = mpnet_model.encode([input for input in inputs])

    # Obtaining similarity
    similarity = list(jnp.squeeze(cos_sim(anchor_emb, inputs_emb)))

    # Returning a Pandas' dataframe
    d = {'inputs': [input for input in inputs],
         'score': [round(similarity[i],3) for i in range(len(similarity))]}
    df = pd.DataFrame(d, columns=['inputs', 'score'])

    return df.sort_values('score', ascending=False)


