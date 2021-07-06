from sentence_transformers import SentenceTransformer, LoggingHandler, models
from seb import Evaluation
import logging
import os
import seb
from transformers import AutoTokenizer, AutoModel
import sys
import types
import torch
import numpy as np
import torch.nn.functional as F

#Use TPU
import torch_xla.core.xla_model as xm
dev = xm.xla_device()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



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
    def __init__(self, model_name, device):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = 128
        self.device = device
        self.model.to(device)

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
        encoded_input = self.tokenizer(sentences, padding='max_length', truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length)

        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device))

        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    #Numpy normalize
    def _normalize(self, x):
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

model_name = sys.argv[1]
logging.info(model_name)

model = SEModel(model_name, device=dev)
eval = Evaluation(datasets_folder='datasets')
eval.run_all(model, tasks=tasks, split='test', output_folder=os.path.join('results', model_name.strip("/").replace("/", "-")))

