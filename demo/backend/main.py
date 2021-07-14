from fastapi import File, Query, FastAPI

import config
import inference
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API of flax-sentence-embeddings."}

@app.get('/similarity')
def get_similarity(anchor: str, inputs: List[str] = Query([]), model: str = 'distilroberta'):
    if model == 'distilroberta':
        return {'dataframe': inference.similarity_distilroberta(anchor, inputs)}

#@app.get("/classify/{code}")
#def get_language(code: str):
#    return {"language": inference.inference(code)}

#@app.get("/code_classification")
#def get_language(code: str = query('')):
#    return {"language": inference.code_classification_inference(code)}


#@app.get("/code_search")
#def get_code(docstring: str = query('')):
#    return {"code": inference.code_search_inference(docstring)}


#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8080)
