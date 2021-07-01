import copy
import gzip
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import BertTokenizer

"""
Implementation of lazy dataloder in PyTorch
"""

class TextIterator:
  def __init__(self, text_iterator, batch_size, num_workers, transform=None):
    self.batch_size = batch_size
    self.iter_number = 0
    self.num_workers = num_workers
    self.text_iterator = text_iterator
    self.transform = transform

  def __iter__(self):
    return self.text_iterator

  def __next__(self):
    if self.iter_number == self.batch_size:
      self.iter_number = 0
      for _ in range(self.batch_size * (self.num_workers - 1)):
        next(self.text_iterator)
    self.iter_number += 1
    answer, question = json.loads(next(self.text_iterator))
    sample = {'question': question, 'answer': answer}
    sample = copy.deepcopy(sample)
    if self.transform:
      sample = self.transform(sample)
    return sample

  def __del__(self):
    self.text_iterator.close()


class TextSimpleIterator:
  def __init__(self, text_iterator, transform=None):
    self.text_iterator = text_iterator
    self.transform = transform

  def __iter__(self):
    return self.text_iterator

  def __next__(self):
    answer, question = json.loads(next(self.text_iterator))
    sample = {'question': question, 'answer': answer}
    if self.transform:
      sample = self.transform(sample)
    return sample

  def __del__(self):
    self.text_iterator.close()


class IterableCorpusDataset(IterableDataset):
  def __init__(self, file_path, batch_size, num_workers, start=0, transform=None):
    self.file_path = file_path
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.start = start
    self.transform = transform

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    dataset_itr = gzip.open(self.file_path, "rb")
    if worker_info is None:
      dataset_itr = gzip.open(self.file_path, "rb")
      for _ in range(self.start):
        next(dataset_itr)
      return TextSimpleIterator(dataset_itr, self.transform)
    else:
      worker_id = worker_info.id
      for _ in range(self.start):
        next(dataset_itr)
      for _ in range(self.batch_size * worker_id):
        next(dataset_itr)
      return TextIterator(dataset_itr, self.batch_size, self.num_workers, self.transform)

