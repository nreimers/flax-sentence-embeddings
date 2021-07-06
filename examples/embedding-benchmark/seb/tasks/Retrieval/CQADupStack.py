from .AbsRetrievalTask import AbsRetrievalTask
import os
from sentence_transformers import util
import numpy as np
import gzip
import json
import logging


class CQADupStack(AbsRetrievalTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/cqadupstack.json.gz'
    local_file_name = 'cqadupstack.json.gz'

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
            self.relevant_docs = {}

            for forum in data:
                self.corpus[forum] = data[forum]['corpus']
                self.queries[forum] = data[forum]['queries']
                self.relevant_docs[forum] = {}
                rel_docs = data[forum]['rel_docs']

                for qid in rel_docs:
                    self.relevant_docs[forum][qid] = set(rel_docs[qid])



    def evaluate(self, model, split='test'):
        assert split == 'test'

        if not self.data_loaded:
            self.load_data()

        scores = {'map@100': None, 'ndcg@10': None, 'forums': {}}
        map_scores = []
        ndcg_scores = []
        for forum in self.corpus:
            scores['forums'][forum] = super().evaluate(model, forum)
            map_scores.append(max(scores['forums'][forum]['cos_sim']['map@k'][100], scores['forums'][forum]['dot_score']['map@k'][100]))
            ndcg_scores.append(max(scores['forums'][forum]['cos_sim']['ndcg@k'][10], scores['forums'][forum]['dot_score']['ndcg@k'][10]))
            logging.info("{} {} MAP@100: {:.4f} \t NDCG@10: {:.4f}".format(forum, " "*(15-len(forum)), map_scores[-1], ndcg_scores[-1]))

        #Compute mean over map scores
        scores['map@100'] = np.mean(map_scores)
        scores['ndcg@10'] = np.mean(ndcg_scores)
        return scores