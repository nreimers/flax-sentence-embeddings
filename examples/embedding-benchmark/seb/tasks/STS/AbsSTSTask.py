from ..AbsTask import AbsTask
from sentence_transformers import util
import os
import gzip
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from abc import abstractmethod

class AbsSTSTask(AbsTask):
    task_type = "sts"

    def __init__(self, **kwargs):
        super(AbsSTSTask, self).__init__(**kwargs)
        self.sts_dataset_path = os.path.join(self.datasets_folder, self.local_file_name)

        self.data_loaded = False
        self.sentences1 = {}
        self.sentences2 = {}
        self.scores = {}


    @property
    @abstractmethod
    def download_url(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def local_file_name(self):
        raise NotImplementedError

    @property
    def min_score(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_score(self):
        raise NotImplementedError

    def load_data(self):
        if self.data_loaded:
            return

        if not os.path.exists(self.sts_dataset_path):
            util.http_get(self.download_url, self.sts_dataset_path)

        self.sentences1 = {split: [] for split in self.description["available_splits"]}
        self.sentences2 = {split: [] for split in self.description["available_splits"]}
        self.scores = {split: [] for split in self.description["available_splits"]}

        with gzip.open(self.sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = (float(row['score']) - self.min_score) / self.max_score  # Normalize score to range 0 ... 1
                self.sentences1[row['split']].append(row['sentence1'])
                self.sentences2[row['split']].append(row['sentence2'])
                self.scores[row['split']].append(score)

        self.data_loaded = True


    def evaluate(self, model, split):
        self.load_data()

        embeddings1 = np.asarray(model.encode(self.sentences1[split.lower()]))
        embeddings2 = np.asarray(model.encode(self.sentences2[split.lower()]))

        gold_scores = self.scores[split.lower()]

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(gold_scores, euclidean_distances)

        return {
            'cosine_pearson': cosine_pearson,
            'cosine_spearman': cosine_spearman,
            'manhatten_pearson': manhatten_pearson,
            'manhatten_spearman': manhatten_spearman,
            'euclidean_pearson': euclidean_pearson,
            'euclidean_spearman': euclidean_spearman,
        }
