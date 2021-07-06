from .AbsSTSTask import AbsSTSTask


class STS15(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/STS15.tsv.gz'
    local_file_name = 'STS15.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "SemEval STS 2015 dataset",
            "reference": "http://alt.qcri.org/semeval2015/task2/",
            "type": self.task_type,
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
        }

