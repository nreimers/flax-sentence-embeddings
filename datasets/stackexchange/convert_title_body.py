"""
Converts the archive from
https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml

To jsonl format

python convert_archives.py input_folder output_folder

Returns (title, body) pairs that pass certain quality checks
"""

import os
import glob
import json
import gzip
import random
from typing import List, Any, IO, Dict
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
import py7zr
import sys

input_folder = sys.argv[1]
output_folder = sys.argv[2]
os.makedirs(output_folder, exist_ok=False)


random.seed(42)

min_title_len = 20
min_body_len = 20
max_body_len = 4096
min_score = 0

large_stackexchange_threshold = 10000   #Stackexchange smaller than this go to a special output file
small_stackexchange_filepath = os.path.join(output_folder, "small_stackexchanges.jsonl")

def parse_posts(f: IO[Any]) -> List[Dict]:
    tree = ET.parse(f)
    posts = tree.getroot()
    pairs = []
    num_questions = 0

    for post in posts:
        data = post.attrib
        if data["PostTypeId"] == "1":  # focus just on questions for now, not answers
            num_questions += 1
            # remove all HTML tags (including links!) and normalize whitespace
            title = re.sub("<.*?>", "", data["Title"]).strip()
            body = re.sub("<.*?>", "", data["Body"]).strip()
            tags_str = data["Tags"]
            tags = re.findall(r"<(.*?)>", tags_str)
            score = int(data["Score"])

            if len(title) < min_title_len or len(body) < min_body_len or len(body) > max_body_len or score < min_score:
                continue

            pairs.append({'texts': [title, body], 'tags': tags})
    print("Questions:", num_questions)
    print("Questions after filter:", len(pairs))
    return pairs


def extract_posts(stack_exchange_file: str) -> List[Dict]:
    with py7zr.SevenZipFile(stack_exchange_file, mode="r") as z:
        fs = z.read(targets=["Posts.xml"])
        if fs is not None and "Posts.xml" in fs:
            posts = parse_posts(fs["Posts.xml"])
            return posts
    return []


def convert_to_jsonl_gz(input_file: str, output_file: str) -> None:
    posts = extract_posts(input_file)
    random.shuffle(posts)
    if len(posts) == 0:
        return

    if len(posts) >= large_stackexchange_threshold:
        fOut = gzip.open(output_file, "wt")
    else:
        fOut = open(small_stackexchange_filepath, "a")

    for post in posts:
        fOut.write(json.dumps(post))
        fOut.write("\n")

    fOut.close()




for filepath in sorted(glob.glob(os.path.join(input_folder, "*.7z")), key=os.path.getsize, reverse=True):
    name = os.path.basename(filepath.strip(".7z"))
    output_path = os.path.join(output_folder, f"{name}.jsonl.gz")
    print(filepath)
    convert_to_jsonl_gz(filepath, output_path)
