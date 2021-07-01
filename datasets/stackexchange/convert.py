from datasets.stackexchange.parser import extract_posts
import json
import gzip


def convert_to_jsonl_gz(input_file: str, output_file: str) -> None:
    posts = extract_posts(input_file)

    with gzip.open(output_file, "w") as f:
        for post in posts:
            json_str = (
                json.dumps({"texts": [post.body, post.title], "tags": post.tags}) + "\n"
            )
            json_bytes = json_str.encode("utf-8")
            f.write(json_bytes)
