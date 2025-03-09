import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import util
from rag import model
import pandas as pd


app = FastAPI()

data_folder = './data'
data = pd.read_csv(os.path.join(data_folder, 'concat_data.csv'))
descriptions = data['cleaned_description'].tolist()
embeddings = model.encode(descriptions, convert_to_tensor=True)


class Query(BaseModel):
    text: str

def find_similar_products(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = cos_scores.topk(top_k)
    return [(data.iloc[int(idx)]['product_name'], score.item()) for idx, score in zip(top_results.indices, top_results.values)]

@app.post("/find_similar")
def find_similar(query: Query):
    similar_products = find_similar_products(query.text)
    return {"similar_products": similar_products}
