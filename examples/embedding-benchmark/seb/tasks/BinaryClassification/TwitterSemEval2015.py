from .AbsBinaryClassificationTask import AbsBinaryClassificationTask


class TwitterSemEval2015(AbsBinaryClassificationTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/TwitterSemEval2015.json.gz'
    local_file_name = 'TwitterSemEval2015.json.gz'


    @property
    def description(self):
        return {
            "name": "TwitterSemEval2015",
            "description": "Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
            "reference": "https://alt.qcri.org/semeval2015/task1/",
            "type": "binary_classfication",
            "available_splits": ["test"],
            "main_score": "ap",
        }


