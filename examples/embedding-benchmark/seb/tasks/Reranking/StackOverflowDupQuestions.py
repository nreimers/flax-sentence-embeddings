from .AbsRankingTask import AbsRankingTask
import os
from sentence_transformers import util
import zipfile
import io

class StackOverflowDupQuestions(AbsRankingTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/StackOverflowDupQuestions.json.gz'
    local_file_name = 'StackOverflowDupQuestions.json.gz'


    @property
    def description(self):
        return {
            "name": "StackOverflowDupQuestions",
            "description": "Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
            "reference": "https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
            "type": "reranking",
            "available_splits": ["train", "dev", "test"],
            "main_score": "map",
        }

