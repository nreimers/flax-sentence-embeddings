from .AbsBinaryClassificationTask import AbsBinaryClassificationTask


class SprintDuplicateQuestions(AbsBinaryClassificationTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/SprintDuplicateQuestions.json.gz'
    local_file_name = 'SprintDuplicateQuestions.json.gz'


    @property
    def description(self):
        return {
            "name": "SprintDuplicateQuestions",
            "description": "Duplicate questions from the Sprint community.",
            "reference": "https://www.aclweb.org/anthology/D18-1131/",
            "type": "binary_classfication",
            "available_splits": ["dev", "test"],
            "main_score": "ap",
        }


