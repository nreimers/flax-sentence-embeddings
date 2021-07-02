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
    tags = []
    num_questions = 0

    mydict = {}
    #Create a dictionary object for every question with Key as the question id and Value as a set of strings containing all details
    for post in posts:
          data = post.attrib
          if data["PostTypeId"] == "1":
              add_data = []
              title = re.sub("<.*?>", "", data["Title"]).strip()
              body = re.sub("<.*?>", "", data["Body"]).strip()
              num_questions += 1
              tags_str = data["Tags"]
              tags.append(re.findall(r"<(.*?)>", tags_str))
              add_data.append(title)
              add_data.append(body)
              score = int(data["Score"])
              if len(title) < min_title_len or len(body) < min_body_len or len(body) > max_body_len or score < min_score: #If length is greater/lesser or score is lesser
                continue
              mydict[int(data["Id"])] = add_data

    #For every answer, checks if it is the best/worst answer for it's corresponding question and changes accordingly.
    for post in posts:
        data = post.attrib
        if data["PostTypeId"] == "2":
            q_id = int(data["ParentId"])
            if mydict[q_id] == None: #If the question was discarded
              continue
            answer = re.sub("<.*?>", "", data["Body"]).strip()
            score = int(data["Score"])
            if len(mydict[q_id]) <= 2: #If this question was encountered first time in the answers list
                mydict[q_id].append(answer) #Adding question for highest score question
                mydict[q_id].append(score)
                mydict[q_id].append(answer) #Adding question for lowest score question
                mydict[q_id].append(score)
            else:
                if mydict[q_id][3] < score: #Comparing if question has higher score than existing question
                    mydict[q_id][3] = score
                    mydict[q_id][2] = answer
                elif mydict[q_id][5] > score: #Comparing if question has lower score than existing question
                    mydict[q_id][5] = score
                    mydict[q_id][4] = answer

    pairs1 = [] #title, body combination
    pairs2 = [] #title, highest_score_answer combination
    pairs3 = [] #title + body, highest_score_answer combination
    pairs4 = [] #title + body, highly_score_answer and low answer combinations
    for post in posts:
        data = post.attrib
        if data["PostTypeId"] == "1":
            pairs1.append({'texts': [mydict[int(data["Id"])][0],[mydict[int(data["Id"])][1]]], 'tags': tags}) #title+body
            if len(mydict[int(data["Id"])])>2:
              pairs2.append({'texts': [mydict[int(data["Id"])][0],[mydict[int(data["Id"])][2]]], 'tags': tags}) #title + highest scored asnwer
              pairs3.append({'texts': [mydict[int(data["Id"])][0] + " " + mydict[int(data["Id"])][1], mydict[int(data["Id"])][2]], 'tags': tags}) #title+body, highest scored answer
              if mydict[int(data["Id"])][3] - mydict[int(data["Id"])][5] >= 100: #If the best and least scored answers have a difference of atleast 100 votes
                pairs4.append({'texts': [mydict[int(data["Id"])][0]+ " " + mydict[int(data["Id"])][1], mydict[int(data["Id"])][2], mydict[int(data["Id"])][4]], 'tags': tags}) #title+body, highloy scored answer, low scored answer 
    
    pairs.append(pairs1)
    pairs.append(pairs2)
    pairs.append(pairs3)
    pairs.append(pairs4) #All pairs are added to a pairs array
    print("Questions:", num_questions)
    print("Questions after filter:", len(pairs1))
    
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
    count = 0 #Used for naming the files according to the combinations 
    for post in posts: 
      random.shuffle(post)
      if len(post) == 0:
          return
        
      if count == 0:
        output_file = os.path.join(output_folder, f"{name}_title_body.jsonl.gz")
      elif count == 1:
        output_file = os.path.join(output_folder, f"{name}_title_highestScoreAnswer.jsonl.gz")
      elif count == 2:
        output_file = os.path.join(output_folder, f"{name}title_body_highestScoreAnswer.jsonl.gz")
      elif count == 3:
        output_file = os.path.join(output_folder, f"{name}title_body_highlyScoredAnswer_lowScoredAnswer.jsonl.gz")
      
      if len(post) >= large_stackexchange_threshold:
          fOut = gzip.open(output_file, "wt")
      else:
          fOut = open(small_stackexchange_filepath, "a")

      for pos in post:
          fOut.write(json.dumps(pos))
          fOut.write("\n")

      fOut.close()
      count = count+1




for filepath in sorted(glob.glob(os.path.join(input_folder, "*.7z")), key=os.path.getsize, reverse=True):
    name = os.path.basename(filepath.strip(".7z"))
    output_path = os.path.join(output_folder, f"{name}.jsonl.gz")
    print(filepath)
    convert_to_jsonl_gz(filepath, output_path)
