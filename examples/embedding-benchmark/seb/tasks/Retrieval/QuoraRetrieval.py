from .AbsRetrievalTask import AbsRetrievalTask


class QuoraRetrieval(AbsRetrievalTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/quora_retrieval.json.gz'
    local_file_name = 'quora_retrieval.json.gz'


    @property
    def description(self):
        return {
            "name": "QuoraRetrieval",
            "description": "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
            "reference": "https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
            "type": "retrieval",
            "available_splits": ["dev", "test"],
            "main_score": "map",
        }
