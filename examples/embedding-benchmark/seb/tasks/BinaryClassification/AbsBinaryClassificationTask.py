import json
from ..AbsTask import AbsTask
from sentence_transformers import util, evaluation
import os
import gzip
import logging
from collections import defaultdict

class AbsBinaryClassificationTask(AbsTask):
    """
    Abstract class for BinaryClassificationTasks

    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise binary classification.
    """

    def __init__(self, **kwargs):
        super(AbsBinaryClassificationTask, self).__init__(**kwargs)
        self.dataset_path = os.path.join(self.datasets_folder, self.local_file_name)
        self.data_loaded = False
        self.data = None

    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        data_split = self.data[split]

        logging.getLogger("sentence_transformers.evaluation.BinaryClassificationEvaluator").setLevel(logging.WARN)
        evaluator = evaluation.BinaryClassificationEvaluator(data_split['sent1'], data_split['sent2'], data_split['labels'])
        scores = evaluator.compute_metrices(model)

        #Compute max
        max_scores = defaultdict(list)
        for sim_fct in scores:
            for metric in ['accuracy', 'f1', 'ap']:
                max_scores[metric].append(scores[sim_fct][metric])

        for metric in max_scores:
            max_scores[metric] = max(max_scores[metric])

        scores['max'] = dict(max_scores)

        return scores

    def load_data(self):
        if self.data_loaded:
            return

        if not os.path.exists(self.dataset_path):
            util.http_get(self.download_url, self.dataset_path)

        with gzip.open(self.dataset_path, 'rt', encoding='utf8') as fIn:
            self.data = json.load(fIn)