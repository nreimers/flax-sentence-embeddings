from .AbsRankingTask import AbsRankingTask
import os
from sentence_transformers import util
import zipfile
import io

class SciDocs(AbsRankingTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/scidocs.json.gz'
    local_file_name = 'scidocs.json.gz'


    @property
    def description(self):
        return {
            "name": "SciDocs",
            "description": "Ranking of related scientific papers based on their title.",
            "reference": "https://allenai.org/data/scidocs",
            "type": "reranking",
            "available_splits": ["dev", "test"],
            "main_score": "map",
        }
