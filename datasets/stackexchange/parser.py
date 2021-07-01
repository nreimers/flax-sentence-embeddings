import io
from typing import List, Any
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass
import py7zr


@dataclass
class StackExchangePost:
    post_id: int
    body: str
    title: str
    tags: List[str]


def parse_posts(f: io.IO[Any]) -> List[StackExchangePost]:
    tree = ET.parse(f)
    posts = tree.getroot()
    pairs: List[StackExchangePost] = []

    for post in tqdm(posts):
        data = post.attrib
        if data["PostTypeId"] == "1":  # focus just on questions for now, not answers
            # remove all HTML tags (including links!) and normalize whitespace
            title = re.sub("\s+", " ", re.sub("<.*?>", "", data["Title"])).strip()
            body = re.sub("\s+", " ", re.sub("<.*?>", "", data["Body"])).strip()
            post_id = int(data["Id"])
            tags_str = data["Tags"]
            tags = re.findall(r"<(.*?)>", tags_str)

            pairs.append(StackExchangePost(post_id, title, body, tags))

    return pairs


def extract_posts(stack_exchange_file: str) -> List[StackExchangePost]:
    with py7zr.SevenZipFile(stack_exchange_file, mode="r") as z:
        fs = z.read(targets=["Posts.xml"])
        if fs is not None:
            posts = parse_posts(fs["Posts.xml"])
            return posts
        else:
            return []
    return []
