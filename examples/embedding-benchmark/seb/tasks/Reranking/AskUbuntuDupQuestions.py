
from .AbsRankingTask import AbsRankingTask


class AskUbuntuDupQuestions(AbsRankingTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/AskUbuntuDupQuestions.json.gz'
    local_file_name = 'AskUbuntuDupQuestions.json.gz'


    @property
    def description(self):
        return {
            "name": "AskUbuntuDupQuestions",
            "description": "AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar",
            "reference": "https://github.com/taolei87/askubuntu",
            "type": "reranking",
            "available_splits": ["test"],
            "main_score": "map",
        }

