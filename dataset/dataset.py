import copy
import gzip
import json
import torch
import math
from torch.utils.data import Dataset, IterableDataset
from numpy.random import choice

"""
Implementation of lazy dataloder in PyTorch
"""

class TextIterator:
  def __init__(self, text_iterator, batch_size, lengths, weights, num_workers,
               transform=None):
    self.batch_size = batch_size
    self.iter_number = 0
    self.num_workers = num_workers
    self.text_iterator = text_iterator
    self.transform = transform
    self.lengths = lengths
    self.weights = weights

  def __iter__(self):
    return self.text_iterator

  def __next__(self):
    sampled_dataset = choice(len(self.lengths), p=self.weights)
    if self.iter_number == self.batch_size:
      self.iter_number = 0
      for _ in range(self.batch_size * (self.num_workers - 1)):
        next(self.text_iterator[sampled_dataset])
    self.iter_number += 1
    answer, question = json.loads(next(self.text_iterator[sampled_dataset]))
    sample = {'question': question, 'answer': answer}
    sample = copy.deepcopy(sample)
    if self.transform:
      sample = self.transform(sample)
    return sample

  def __del__(self):
    self.text_iterator.close()


class TextSimpleIterator:
  def __init__(self, text_iterator, lengths, weights, transform=None):
    self.text_iterator = text_iterator
    self.transform = transform
    self.lengths = lengths
    self.weights = weights

  def __iter__(self):
    return self.text_iterator

  def __next__(self):
    sampled_dataset = choice(len(self.lengths), p=self.weights)
    answer, question = json.loads(next(self.text_iterator[sampled_dataset]))
    sample = {'question': question, 'answer': answer}
    if self.transform:
      sample = self.transform(sample)
    return sample

  def __del__(self):
    self.text_iterator.close()


class IterableCorpusDataset(IterableDataset):
  def __init__(self, file_paths, lengths, batch_size, num_workers, K=2097152,
               T=2, transform=None):
    self.file_paths = file_paths
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transform = transform
    self.lengths = lengths
    lengths_sum = sum(lengths)
    weights = [min(length, K) / lengths_sum for length in lengths]
    weights = list(map(lambda x: math.pow(x, (1 / T)), weights))
    tot_weights = sum(weights)
    self.weights = list(map(lambda x: x / tot_weights, weights))

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    dataset_iterators = {}
    for idx, path in enumerate(self.file_paths):
        dataset_iterators[idx] = gzip.open(path, "rb")
    if worker_info is None:
      return TextSimpleIterator(dataset_iterators, self.lengths,
                                self.weights, self.transform)
    else:
      return TextIterator(dataset_iterators, self.batch_size, self.lengths,
                          self.weights, self.num_workers, self.transform)