from .AbsClusteringTask import AbsClusteringTask


class StackExchangeClustering(AbsClusteringTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/stackexchange-clustering.json.gz'
    local_file_name = 'stackexchange-clustering.json.gz'

    @property
    def description(self):
        return {
            "name": "StackExchangeClustering",
            "description": "Clustering of titles from 121 stackexchanges. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "clustering",
            "available_splits": ["dev", "test"],
            "main_score": "v_measure",
        }



