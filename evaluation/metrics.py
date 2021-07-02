from os import name
import numpy as np
from tqdm import tqdm

def recall_k(n,sim_func,contexts, responses):
    """
    Recall 1-of-N metric as used in conveRT paper. 

    Recall@k takes N responses to the given conversational context, where only one response is relevant. 
    It indicates whether the relevant response occurs in the top k ranked candidate responses. 
    The 1-of-N metric is obtained when k=1. This effectively means that, for each query, 
    we indicate if the correct response is the top ranked response among N candidates. 
    The final score is the average across all queries

    from https://github.com/PolyAI-LDN/conversational-datasets/blob/master/baselines/run_baseline.py

    :param n: Number of response candidates to be passed to a context for retrieval
    :param contexts: context embeddings - shape (num_embs,emb_dim)
    :param responses: response embeddings - shape (num_embs,emb_dim)
    :param sim_func: similarity function - cosine or dot product, which returns an similarity scores of shape (num_embs,num_embs)
    :return: recall score 

    """
    accuracy_numerator = 0.0
    accuracy_denominator = 0.0
    for i in tqdm(range(0, len(contexts), n)):
        context_batch = contexts[i:i + n]
        responses_batch = responses[i:i + n]
        if len(context_batch) != n:
            break

        # Shuffle the responses.
        permutation = np.arange(n)
        np.random.shuffle(permutation)
        context_batch_shuffled = [context_batch[j] for j in permutation]

        predictions = np.argmax(sim_func(context_batch_shuffled, responses_batch),axis=1)
        accuracy_numerator += np.equal(predictions, permutation).mean()
        accuracy_denominator += 1.0

    accuracy = 100 * accuracy_numerator / accuracy_denominator
    return accuracy