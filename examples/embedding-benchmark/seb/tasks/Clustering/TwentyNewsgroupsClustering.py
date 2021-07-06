from .AbsClusteringTask import AbsClusteringTask



class TwentyNewsgroupsClustering(AbsClusteringTask):
    download_url = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/seb/datasets/20NewsgroupsClusters.json.gz'
    local_file_name = '20NewsgroupsClusters.json.gz'

    @property
    def description(self):
        return {
            "name": "20NewsgroupsClustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "clustering",
            "available_splits": ["test"],
            "main_score": "v_measure",
        }



