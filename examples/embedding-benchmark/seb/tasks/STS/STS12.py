from .AbsSTSTask import AbsSTSTask


class STS12(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/STS12.tsv.gz'
    local_file_name = 'STS12.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "SemEval STS 2012 dataset.",
            "reference": "https://www.aclweb.org/anthology/S12-1051.pdf",
            "type": self.task_type,
            "available_splits": ["train", "test"],
            "main_score": "cosine_spearman",
        }

