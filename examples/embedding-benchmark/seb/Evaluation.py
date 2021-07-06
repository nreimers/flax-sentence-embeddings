from .tasks import *
import logging
import json
import os

class Evaluation:
    def __init__(self, datasets_folder='datasets'):
        self.datasets_folder = datasets_folder
        self.selected_tasks = None

    @property
    def available_tasks_classes(self):
        return [
            #TODO
        ]

    @property
    def available_tasks(self):
        return [taskclass(datasets_folder=self.datasets_folder) for taskclass in self.available_tasks_classes]

    @staticmethod
    def run_all(model, tasks, split, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for task in tasks:
            name = task.description['name']
            if split in task.description['available_splits']:
                results_filepath = os.path.join(output_folder, name+".json")

                if os.path.exists(results_filepath):
                    logging.info("Skip {}, results file: {} already exsists".format(name, results_filepath))
                    continue

                logging.info("Start evaluation on task: "+name)
                results = task.evaluate(model, split=split)
                logging.info("Evaluation results on task "+name)
                logging.info(results)
                with open(results_filepath, 'w') as fOut:
                    json.dump(results, fOut, indent=3, sort_keys=True)