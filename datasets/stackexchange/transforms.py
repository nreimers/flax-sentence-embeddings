"""
Converts the archive from
https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml

To jsonl format

python convert_archives.py input_folder output_folder

Returns the following cominations that pass certain quality checks:
    -> title, body combination
    -> title, highest_score_answer combination
    -> title + body, highest_score_answer combination
    -> title + body, highly_score_answer and low answer combinations
"""
#=====================================================================================
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

title_answer_folder = output_folder + "/TitleAnswer/"
os.makedirs(title_answer_folder, exist_ok=False)

titlebody_answer_folder = output_folder + "/TitleBodyAnswer/"
os.makedirs(titlebody_answer_folder, exist_ok=False)

titlebody_best_worst_answer_folder = output_folder + "/TitleBodyBestWorstAnswer/"
os.makedirs(titlebody_best_worst_answer_folder, exist_ok=False)

random.seed(42)

min_title_len = 20
min_body_len = 20
max_body_len = 4096
min_score = 0
#=======================================================================================
#Creates a dictionary object for every question with Key as the question id and Value as a set of strings containing all details
def create_dict_for_questions(posts):
    num_questions = 0
    mydict = {}
    for post in posts:
        data = post.attrib
        if data["PostTypeId"] == "1":
            add_data = []
            title = re.sub("<.*?>", "", data["Title"]).strip()
            body = re.sub("<.*?>", "", data["Body"]).strip()
            num_questions += 1
            add_data.append(title)
            add_data.append(body)
            score = int(data["Score"])
            if len(title) < min_title_len or len(body) < min_body_len or len(body) > max_body_len or score < min_score: #If length is greater/lesser or score is lesser
                continue
            mydict[int(data["Id"])] = add_data
    
    #For every answer, checks for the best/worst answer for it's corresponding question and changes accordingly.
    for post in posts:
        data = post.attrib
        if data["PostTypeId"] == "2":
            q_id = int(data["ParentId"])
            if (q_id in mydict.keys()): #If the question was discarded
                answer = re.sub("<.*?>", "", data["Body"]).strip()
                score = int(data["Score"])
                if len(mydict[q_id]) <= 2: #If this question was encountered first time in the answers list
                    mydict[q_id].append(answer) #Adding question for maximum score question
                    mydict[q_id].append(score)
                    mydict[q_id].append(answer) #Adding question for minimum score question
                    mydict[q_id].append(score)
                else:
                    if mydict[q_id][3] < score: #Comparing if question has higher score than existing question
                        mydict[q_id][3] = score
                        mydict[q_id][2] = answer
                    elif mydict[q_id][5] > score: #Comparing if question has lower score than existing question
                        mydict[q_id][5] = score
                        mydict[q_id][4] = answer
    return mydict
#==========================================================================================
def extract_title_body(mydict):
    pairs = [] #title_body combination
    for key in mydict:
        pairs.append([ mydict[key][0],mydict[key][1] ]) #title+body
    return pairs
    
def extract_title_highestscored(mydict):
    pairs = [] #title_highestScoreAnswer
    for key in mydict:
        if len(mydict[key])>2:
            pairs.append([ mydict[key][0],mydict[key][2] ]) #title + highest scored
    return pairs

def extract_title_body_highscore(mydict):
    pairs = [] #title_body_highestScoreAnswer
    for key in mydict:
        if len(mydict[key])>2:
            pairs.append([ mydict[key][0]+ " " +mydict[key][1], mydict[key][2] ]) #title+body, highest scored
    return pairs
    
def extract_title_body_highscore_lowscore(mydict):
    pairs = [] #title_body_highestScoreAnswer_lowlyScoredAnswer
    for key in mydict:
        if len(mydict[key])>5:
            if ((mydict[key][3] - mydict[key][5] >= 100) or (mydict[key][5]<0 )): #If the best and least answers have a difference of 100 votes
                pairs.append([ mydict[key][0]+ " " +mydict[key][1], mydict[key][2], mydict[key][4] ]) #title+body, high scored,least scored 
    return pairs
#===========================================================================================    
def parse_posts(f: IO[Any]) -> List[Dict]:
    tree = ET.parse(f)
    posts = tree.getroot()
    pairs = []
    tags = []
    num_questions = 0
    mydict = create_dict_for_questions(posts)
    return mydict
#============================================================================================    
def extract_posts(stack_exchange_file: str) -> List[Dict]:
    with py7zr.SevenZipFile(stack_exchange_file, mode="r") as z:
        fs = z.read(targets=["Posts.xml"])
        if fs is not None and "Posts.xml" in fs:
            posts = parse_posts(fs["Posts.xml"])
            return posts
    return []
#=============================================================================================
def convert_to_jsonl_gz(input_file: str, output_file: str) -> None:
    mydict = extract_posts(input_file)  
    #save title_body combination
    #posts = extract_title_body(mydict)
    #random.shuffle(posts)
    #output_file = os.path.join(output_folder, f"{name}_title_body.jsonl.gz")
    #if len(posts) == 0:
    #    return
    #fOut = gzip.open(output_file, "wt")
    #for post in posts:
    #    fOut.write(json.dumps(post))
    #    fOut.write("\n")
    #fOut.close()
    #save title_highestScoreAnswer combination
    posts = extract_title_highestscored(mydict)
    random.shuffle(posts)
    output_file = os.path.join(title_answer_folder, f"{name}.jsonl.gz")
    if len(posts) == 0:
        return
    fOut = gzip.open(output_file, "wt")
    for post in posts:
        fOut.write(json.dumps(post))
        fOut.write("\n")
    fOut.close()
    #save title_body_highestScoreAnswer combination
    posts = extract_title_body_highscore(mydict)
    random.shuffle(posts)
    output_file = os.path.join(titlebody_answer_folder, f"{name}.jsonl.gz")
    if len(posts) == 0:
        return
    fOut = gzip.open(output_file, "wt")    
    for post in posts:
        fOut.write(json.dumps(post))
        fOut.write("\n")
    fOut.close()
    #save title_body_highestScoreAnswer_lowlyScoredAnswer combination
    posts = extract_title_body_highscore_lowscore(mydict)
    random.shuffle(posts)
    output_file = os.path.join(titlebody_best_worst_answer_folder, f"{name}.jsonl.gz")
    if len(posts) == 0:
        return
    fOut = gzip.open(output_file, "wt")
    for post in posts:
        fOut.write(json.dumps(post))
        fOut.write("\n")
    fOut.close()
#========================================================================================================
for filepath in sorted(glob.glob(os.path.join(input_folder, "*.7z")), key=os.path.getsize, reverse=True):
    name = os.path.basename(filepath.strip(".7z"))
    output_path = os.path.join(output_folder, f"{name}.jsonl.gz")
    print(filepath)
    convert_to_jsonl_gz(filepath, output_path)
