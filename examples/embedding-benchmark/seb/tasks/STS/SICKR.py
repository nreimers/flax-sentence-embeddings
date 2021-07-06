from .AbsSTSTask import AbsSTSTask


class SICKR(AbsSTSTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/sick-r.tsv.gz'
    local_file_name = 'sick-r.tsv.gz'
    min_score = 1
    max_score = 5

    @property
    def description(self):
        return {
            "name": "SICK-R",
            "description": "Semantic Textual Similarity SICK-R dataset as described here:",
            "reference": "https://www.aclweb.org/anthology/S14-2001.pdf",
            "type": self.task_type,
            "available_splits": ["train", "dev", "test"],
            "main_score": "cosine_spearman",
        }

