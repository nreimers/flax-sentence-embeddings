from .AbsSTSTask import AbsSTSTask

class BIOSSES(AbsSTSTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/BIOSSES.tsv.gz'
    local_file_name = 'BIOSSES.tsv.gz'
    min_score = 0
    max_score = 4

    @property
    def description(self):
        return {
            "name": "BIOSSES",
            "description": "Biomedical Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": self.task_type,
            "available_splits": ["test"],
            "main_score": "cosine_spearman",
        }

