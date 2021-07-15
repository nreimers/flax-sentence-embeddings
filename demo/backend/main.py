from fastapi import Query, FastAPI

import config
import inference
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API of flax-sentence-embeddings."}

@app.get('/similarity')
def get_similarity(anchor: str, inputs: List[str] = Query([]), model: str = 'distilroberta'):
    return {'dataframe': inference.text_similarity(anchor, inputs, model)}


#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8080)
