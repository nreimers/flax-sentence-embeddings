import torch
from torch import Tensor


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def mean_pooling(model_output, attention_mask):
    """
    Returns mean pooled embeddings from the last layer of a PyTorch based HuggingFace Transformer model.
    """
    embeddings = model_output[0]
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def max_pooling(model_output, attention_mask):
    """
    Returns max pooled embeddings from the last layer of a PyTorch based HuggingFace Transformer model.
    """
    embeddings = model_output[0]
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()    
    return torch.max(embeddings * attention_mask_expanded, 1).values

def cls_pooling(model_output):
    """
    Returns [CLS] token embedding from the last layer of a PyTorch based HuggingFace Transformer model.
    """
    return model_output[0][:, 0] # 1st token is the [CLS] token