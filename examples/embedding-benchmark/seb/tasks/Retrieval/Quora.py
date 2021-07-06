from .AbsRetrievalTask import AbsRetrievalTask
import os
from sentence_transformers import util
import numpy as np
import gzip
import json
import logging
from collections import defaultdict

class Quora(AbsRetrievalTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/cqadupstack.json.gz'
    local_file_name = 'cqadupstack.json.gz'

    def __init__(self, **kwargs):
        super(CQADupStack, self).__init__(**kwargs)
        self.dataset_path = os.path.join(self.datasets_folder, self.local_file_name)
        self.data_loaded = False
        self.corpus = None
        self.queries = None
        self.relevant_docs = None



    @property
    def description(self):
        return {
            "name": "CQADupStack",
            "description": "CQADupStack is a benchmark dataset for community question-answering (cQA) research. Here, we use the task of finding duplicate questions for 12 different StackExchange communities.",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "retrieval",
            "available_splits": ["test"],
            "main_score": "map",
        }

    def load_data(self):
        if self.data_loaded:
            return

        if not os.path.exists(self.dataset_path):
            util.http_get(self.download_url, self.dataset_path)

        with gzip.open(self.dataset_path, 'rt', encoding='utf8') as fIn:
            data = json.load(fIn)
            self.corpus = {}
            self.queries = {}
            self.relevant_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            for forum in data:
                self.corpus[forum] = data[forum]['corpus']
                self.queries[forum] = data[forum]['queries']
                rel_docs = data[forum]['rel_docs']

                for qid in rel_docs:
                    for pid in rel_docs[qid]:
                        self.relevant_docs[forum][qid][pid] = 1


    def evaluate(self, model, split='test'):
        assert split == 'test'

        if not self.data_loaded:
            self.load_data()

        scores = {'map@100': None, 'ndcg@10': None, 'forums': {}}
        map_scores = []
        ndcg_scores = []
        for forum in self.corpus:
            scores['forums'][forum] = super().evaluate(model, forum)
            map_scores.append(scores['forums'][forum]['map@100'])
            ndcg_scores.append(scores['forums'][forum]['ndcg@10'])
            logging.info("{} {} MAP@100: {:.4f} \t NDCG@10: {:.4f}".format(forum, " "*(15-len(forum)), scores['forums'][forum]['map@100'], scores['forums'][forum]['ndcg@10']))

        #Compute mean over map scores
        scores['map@100'] = np.mean(map_scores)
        scores['ndcg@10'] = np.mean(ndcg_scores)
        return scores