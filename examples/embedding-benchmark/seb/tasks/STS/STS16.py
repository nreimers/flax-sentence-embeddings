from .AbsSTSTask import AbsSTSTask


class STS16(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/STS16.tsv.gz'
    local_file_name = 'STS16.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "SemEval STS 2016 dataset",
            "reference": "http://alt.qcri.org/semeval2016/task1/",
            "type": self.task_type,
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
        }

