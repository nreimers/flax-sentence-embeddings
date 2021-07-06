import jax
from sentence_transformers import SentenceTransformer, LoggingHandler, models
from seb import Evaluation
import logging
import os
import seb
from transformers import AutoTokenizer, FlaxAutoModel
import sys
import torch
import numpy as np

import sys
sys.path.append("../..")

from trainer.utils.ops import normalize_L2, mean_pooling

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info("JAX devices:", jax.devices())

tasks = [
    seb.tasks.BinaryClassification.SprintDuplicateQuestions(datasets_folder='datasets'),
    seb.tasks.BinaryClassification.TwitterSemEval2015(datasets_folder='datasets'),
    seb.tasks.BinaryClassification.TwitterURLCorpus(datasets_folder='datasets'),
    seb.tasks.STS.STSbenchmark(datasets_folder='datasets'),
    seb.tasks.STS.SICKR(datasets_folder='datasets'),
    seb.tasks.STS.BIOSSES(datasets_folder='datasets'),
    seb.tasks.Reranking.AskUbuntuDupQuestions(datasets_folder='datasets'),
    seb.tasks.Reranking.StackOverflowDupQuestions(datasets_folder='datasets'),
    seb.tasks.Reranking.SciDocs(datasets_folder='datasets'),
    seb.tasks.Retrieval.QuoraRetrieval(datasets_folder='datasets'),
    seb.tasks.Retrieval.CQADupStack(datasets_folder='datasets'),
    seb.tasks.Clustering.TwentyNewsgroupsClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.StackExchangeClustering(datasets_folder='datasets'),
    seb.tasks.Clustering.RedditClustering(datasets_folder='datasets'),
]


logging.info("System to evaluate: {}".format(len(sys.argv[1:])))


class SEModel:
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
logging.info(model_name)

model = SEModel(model_name)
eval = Evaluation(datasets_folder='datasets')
eval.run_all(model, tasks=tasks, split='test', output_folder=os.path.join('results', model_name.strip("/").replace("/", "-")))

