"""
Downloads all archive files for stackexchange from: https://archive.org/download/stackexchange

Requires:
pip install sentence-transformers

for the download
"""

from sentence_transformers import util
import os

with open('download_archive_file_list.tsv') as fIn:
    for line in fIn:
        name = line.strip().split("\t")[0]
        output_path = os.path.join("archive", name)
        if os.path.exists(output_path):
            continue
            
        if name.endswith('.7z') and '.meta.' not in name:
            print("Download:", name)
            util.http_get("https://archive.org/download/stackexchange/"+name, output_path)