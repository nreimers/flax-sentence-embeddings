# flax-sentence-embeddings

This repository will be used to share code for the Flax / JAX community event to train sentence embeddings on 1B+ training pairs.

You can add your code by creating a pull request.


## Dataloading

### Dowload data

You can download the data using this basic python script at the root of the project. 
Download should be completed in about 20 minutes given your connection speed. Total size on disk is arround 25G. 

```bash
python dataset/download_data.py --dataset_list=dataset/datasets_list.tsv --data_path=PATH_TO_STORE_DATASETS
```
On a different note:

There is another directory called `dataset_list`, which contains a subdirectory called `stackexchange`. This subdirectory contains the script to download the compressed stackexchange `xml` files from the internet archive. Once downloaded, these compressed stackexchange xml files need to be converted to the `jsonl` format for training purpose. To transform, use the the `datasets/stackexchange/transforms.py` script, which generates the required input data format for training. A small file restructuring or cleanup is required, as the directories `dataset`, `datasets` and `dataset_list` could be confusing.

### Dataloading

First implementation of the dataloader takes as input a single `jsonl.gz` file. 
It creates a pointer on the file such that samples are loaded one by one.
The implementation is based on `torch` standard `Dataloader` and `Dataset` classes.
The class supports `num_worker>0` such that data loading is done in a background process on the CPU, i.e. the data is loaded and tokenized in parallel to training the network. 
This avoid to create a bottleneck from I/O and tokenization. The implementation currently return `{'anchor': '...,' 'positive': '...'}`

```
from dataset.dataset import IterableCorpusDataset

corpus_dataset = IterableCorpusDataset(
  file_path=os.path.join(PATH_TO_STORE_DATASETS, 'stackexchange_duplicate_questions_title_title.json.gz'), 
  batch_size=2,
  num_workers=2, 
  transform=None)

corpus_dataset_itr = iter(corpus_dataset)
next(corpus_dataset_itr)

# {'anchor': 'Can anyone explain all these Developer Options?',
#  'positive': 'what is the advantage of using the GPU rendering options in Android?'}

def collate(batch_input_str):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch = {'anchor': tokenizer.batch_encode_plus([b['anchor'] for b in batch_input_str], pad_to_max_length=True),
             'positive': tokenizer.batch_encode_plus([b['positive'] for b in batch_input_str], pad_to_max_length=True)}
    return batch

corpus_dataloader = DataLoader(
  corpus_dataset,
  batch_size=2,
  num_workers=2,
  collate_fn=collate,
  pin_memory=False,
  drop_last=True,
  shuffle=False)

print(next(iter(corpus_dataloader)))

# {'anchor': {'input_ids': [[101, 4531, 2019, 2523, 2090, 2048, 4725, 1997, 2966, 8830, 1998, 1037, 7142, 8023, 102, 0, 0, 0], [101, 1039, 1001, 10463, 5164, 1061, 2100, 2100, 24335, 26876, 11927, 4779, 4779, 2102, 2000, 3058, 7292, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}, 'positive': {'input_ids': [[101, 1045, 2031, 2182, 2007, 2033, 1010, 2048, 4725, 1997, 8830, 1025, 1037, 3115, 2729, 4118, 1010, 1998, 1037, 17009, 8830, 1012, 2367, 3633, 4374, 2367, 4118, 1010, 2049, 2035, 18154, 11095, 1012, 1045, 2572, 2667, 2000, 2424, 1996, 2523, 1997, 1996, 17009, 8830, 1998, 1037, 1005, 2092, 2108, 3556, 1005, 2029, 2003, 1037, 15973, 3643, 1012, 2054, 2003, 1996, 2190, 2126, 2000, 2424, 2151, 8924, 1029, 1041, 1012, 1043, 1012, 8833, 6553, 26237, 2944, 1029, 102], [101, 1045, 2572, 2667, 2000, 10463, 1037, 5164, 3058, 2046, 1037, 4289, 2005, 29296, 3058, 7292, 1012, 1996, 4289, 2003, 2066, 1024, 1000, 2297, 2692, 20958, 2620, 17134, 19317, 19317, 1000, 1045, 2228, 2023, 1041, 16211, 4570, 2000, 1061, 2100, 2100, 24335, 26876, 11927, 4779, 4779, 2102, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}}

```   
=======

## Installation

### Poetry

A Poetry toml is provided to manage dependencies in a virtualenv. Check https://python-poetry.org/

Once you've installed poetry, you can connect to virtual env and update dependencies:
 
```
poetry shell
poetry update
poetry install
```

### requirements.txt

Someone on your platform should generate it once with following command.

```
poetry export -f requirements.txt --output requirements.txt
```

### Rust compiler for hugginface tokenizers

- Hugginface tokenizers require a Rust compiler so install one.

### custom libs

- If you want a specific version of any library, edit the pyproject.toml, add it and/or replace "*" by it.

## Running Tests

Call this in the project folder to execute unit tests.

```
python -m unittest discover -s tests
```



