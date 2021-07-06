from .AbsSTSTask import AbsSTSTask


class STSbenchmark(AbsSTSTask):
    download_url = 'https://sbert.net/datasets/stsbenchmark.tsv.gz'
    local_file_name = 'stsbenchmark.tsv.gz'
    min_score = 0
    max_score = 5

    @property
    def description(self):
        return {
            "name": "STSbenchmark",
            "description": "Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
            "reference": "http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark",
            "type": self.task_type,
            "available_splits": ["train", "dev", "test"],
            "main_score": "cosine_spearman",
        }

