from .AbsSTSTask import AbsSTSTask


class STS14(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/STS14.tsv.gz'
    local_file_name = 'STS14.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "SemEval STS 2014 dataset. Currently only the English dataset",
            "reference": "http://alt.qcri.org/semeval2014/task10/",
            "type": self.task_type,
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
        }

