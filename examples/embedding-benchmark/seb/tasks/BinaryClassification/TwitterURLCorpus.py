from .AbsBinaryClassificationTask import AbsBinaryClassificationTask


class TwitterURLCorpus(AbsBinaryClassificationTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/TwitterURLCorpus.json.gz'
    local_file_name = 'TwitterURLCorpus.json.gz'


    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "type": "binary_classfication",
            "available_splits": ["test"],
            "main_score": "ap",
        }


