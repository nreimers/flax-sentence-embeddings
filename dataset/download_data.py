import os, sys
import argparse
import pandas as pd
import urllib.request

def download_dataset(url, file_name):
  urllib.request.urlretrieve(url, file_name)
  return

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='Dowload raw coropora given dataset list.')
  parser.add_argument('--dataset_list')
  parser.add_argument('--data_path')
  args = parser.parse_args(args=sys.argv[1:])
  
  datasets = pd.read_csv(
    args.dataset_list,
    index_col=0,
    sep='\t',
    dtype={
      'Description': str,
      'Size (#Pairs)': str,
      'Performance': float,
      'Download link': str,
      'Source': str})
  datasets['Size (#Pairs)'] = datasets['Size (#Pairs)'].str.replace(',', '').astype(int)
  datasets = datasets.to_dict(orient='index')
  
  print('Downloading {:,} dataset into {}.'.format(len(datasets), args.data_path))
  
  for d in datasets.keys():
    print('Downloading dataset {} ({:,} pairs) ... '.format(d, datasets[d]['Size (#Pairs)']), end='', flush=True)
    download_dataset(
      datasets[d]['Download link'],
      os.path.join(os.path.abspath(args.data_path), d + '.json.gz'))
    print('\033[32m' + 'Done' + '\033[0m')