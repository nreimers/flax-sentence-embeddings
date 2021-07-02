"""
This is an example how to train with sentence-transformers.

It trains the model just for 2k steps using equal weighting of all provided dataset files.

Run:
python training.py exp-name file1.jsonl.gz [file2.jsonl.gz] ...

"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from MultiDatasetDataLoader import MultiDatasetDataLoader
from shutil import copyfile
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

exp_name = sys.argv[1]


model_name = 'nreimers/MiniLM-L6-H384-uncased'
batch_size_pairs = 256
batch_size_triplets = 256 
steps_per_epoch = 2000

num_epochs = 1
max_seq_length = 128
use_amp = True
warmup_steps = 500

#####

output_path = 'output/training_data_benchmark-{}-norm-{}'.format(model_name.replace("/", "-"), exp_name)
logging.info("Output: "+output_path)
if os.path.exists(output_path):
    exit()


# Write train script to output path
os.makedirs(output_path, exist_ok=True)

train_script_path = os.path.join(output_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

## SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
norm = models.Normalize()
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, norm])

datasets = []
for filepath in sys.argv[2:]:
    filepath = filepath.strip()
    dataset = []
    
    with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())
            
            if not isinstance(data, dict):
                data = {'guid': None, 'texts': data}
           
            dataset.append(InputExample(guid=data.get('guid', None), texts=data['texts']))
            if len(dataset) >= (steps_per_epoch * batch_size_pairs * 2):
                break

    datasets.append(dataset)
    logging.info("{}: {}".format(filepath, len(dataset)))


# Special data loader to load from multiple datasets
train_dataloader = MultiDatasetDataLoader(datasets, batch_size_pairs=batch_size_pairs, batch_size_triplets=batch_size_triplets)


# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model, scale=20, similarity_fct=util.dot_score)



# Configure the training
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=None,
          epochs=1,
          warmup_steps=warmup_steps,
          steps_per_epoch=steps_per_epoch,
          scheduler='warmupconstant',   #Remove this line when you train on larger datasets. After warmup, LR will be constant
          use_amp=use_amp
          )


model.save(output_path)