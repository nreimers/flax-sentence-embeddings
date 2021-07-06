from .AbsSTSTask import AbsSTSTask


class STS13(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/STS13.tsv.gz'
    local_file_name = 'STS13.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "SemEval STS 2013 dataset.",
            "reference": "https://www.aclweb.org/anthology/S13-1004/",
            "type": self.task_type,
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
        }

