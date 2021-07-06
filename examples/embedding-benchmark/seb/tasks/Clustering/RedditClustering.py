from .AbsClusteringTask import AbsClusteringTask
import os
from sentence_transformers import util
import numpy as np
import gzip
import json
import random

class RedditClustering(AbsClusteringTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/reddit-clustering.json.gz'
    local_file_name = 'reddit-clustering.json.gz'



    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "description": "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "clustering",
            "available_splits": ["dev", "test"],
            "main_score": "v_measure",
        }

