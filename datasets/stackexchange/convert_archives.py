"""
Downloads all archive files for stackexchange from: https://archive.org/download/stackexchange

Requires:
pip install sentence-transformers

for the download
"""

import os
import glob
import sys

from datasets.stackexchange.convert import convert_to_jsonl_gz

input_path_files = os.path.join("data/archive", "*.7z")

files = glob.glob(input_path_files)

output_path = "data/jsonl"
os.makedirs(output_path, exist_ok=True)

for f in files:
    name = os.path.basename(f).strip(".7z")
    print(name)
    convert_to_jsonl_gz(f, os.path.join(output_path, f"{name}.jsonl.gz"))
