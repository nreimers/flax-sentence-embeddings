#import gzip
#import json

MODEL_PATH = './models/'

MODELS_ID = dict(code_classification = 'huggingface/CodeBERTa-language-id',
                 code_search = 'flax-sentence-embeddings/st-codesearch-distilroberta-base',
                 distilroberta = 'flax-sentence-embeddings/st-codesearch-distilroberta-base',
                 mpnet = 'flax-sentence-embeddings/all_datasets_v3_mpnet-base')


# Opening dataset from code-search-net
#with gzip.open("data/codesearchnet.jsonl.gz", "r") as f:
#   data = f.read()
#   code_search_net = [json.loads(jline) for jline in data.splitlines()]

# Opening the embedded docstrings from code-search-net
#with gzip.open('data/descriptions_emb.pt.gz', 'r') as f:
#    embedded_docstrings_code_search = f.read()

#DATASETS = dict(
#    code_search = dict(
#        docstrings_embedding =embedded_docstrings_code_search,
#        docstrings_codes_lists = code_search_net)
#    )
