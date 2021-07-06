import json
from ..AbsTask import AbsTask
from sentence_transformers import util, evaluation
import os
import gzip
from collections import defaultdict
import numpy as np

class AbsRankingTask(AbsTask):
    """
    Abstract class for re-ranking experiments.

    Child-classes must implement the following properties:
    self.sentences = {}         #id => sentence
    self.queries = {'dev': List[str], 'test': List[str]}
    self.candidates = {'dev': List[str], 'test': List[str]}
    self.candidate_ids = {'dev': List[key], 'test': List[key]}  #Ids of the candidates
    self.similar_ids = {'dev': List[key], 'test': List[key]}    #Which of the candidate ids are relevant
    """

    def __init__(self, **kwargs):
        super(AbsRankingTask, self).__init__(**kwargs)
        self.dataset_path = os.path.join(self.datasets_folder, self.local_file_name)
        self.data_loaded = False
        self.data = None


    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        data_split = self.data[split]

        if isinstance(data_split, dict):
            sub_scores = {}
            scores = defaultdict(list)
            for name, eval_data in data_split.items():
                rr_evaluator = evaluation.RerankingEvaluator(eval_data, show_progress_bar=False)
                sub_scores[name] = rr_evaluator.compute_metrices(model)

                for eval_metric in sub_scores[name]:
                    scores[eval_metric].append(sub_scores[name][eval_metric])

            ##Compute mean over all subsplits
            for eval_metric in scores:
                scores[eval_metric] = np.mean(scores[eval_metric])

            scores['sub_scores'] = sub_scores

        else:
            rr_evaluator = evaluation.RerankingEvaluator(data_split, show_progress_bar=False)
            scores = rr_evaluator.compute_metrices(model)

        return dict(scores)

    def load_data(self):
        if self.data_loaded:
            return

        if not os.path.exists(self.dataset_path):
            util.http_get(self.download_url, self.dataset_path)

        with gzip.open(self.dataset_path, 'rt', encoding='utf8') as fIn:
            self.data = json.load(fIn)

