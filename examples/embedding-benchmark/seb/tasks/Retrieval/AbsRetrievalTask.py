import logging

from ..AbsTask import AbsTask
from sentence_transformers import util, evaluation
import numpy as np
import os
import json
import gzip
import logging
class AbsRetrievalTask(AbsTask):
    """
    Abstract class for re-ranking experiments.

    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """
    def __init__(self, **kwargs):
        super(AbsRetrievalTask, self).__init__(**kwargs)

        self.dataset_path = os.path.join(self.datasets_folder, self.local_file_name)
        self.data_loaded = False
        self.corpus = None
        self.queries = None
        self.relevant_docs = None

    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        corpus = self.corpus[split] #qid => query
        queries = self.queries[split] #cid => doc
        relevant_docs = self.relevant_docs[split] #qid => Set[cid]

        #Convert lists to sets
        for doc_id in relevant_docs:
            relevant_docs[doc_id] = set(relevant_docs[doc_id])

        logging.getLogger("sentence_transformers.evaluation.InformationRetrievalEvaluator").setLevel(logging.WARN)
        evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs)
        scores = evaluator.compute_metrices(model)
        return scores

    def load_data(self):
        if self.data_loaded:
            return

        if not os.path.exists(self.dataset_path):
            util.http_get(self.download_url, self.dataset_path)

        with gzip.open(self.dataset_path, 'rt', encoding='utf8') as fIn:
            data = json.load(fIn)
            self.corpus = data['corpus']
            self.queries = data['queries']
            self.relevant_docs = data['relevant_docs']