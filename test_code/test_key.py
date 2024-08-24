from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
import torch
import uuid

collection_name = "yasuda_ver.1.2"
query = "女性に人気はありますか？"
# モデルとトクナイザーの読み込み
model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

qdrant_client = QdrantClient("http://localhost:6333")

def encode_text(text: str):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

query_vector = encode_text(query)
search_result = qdrant_client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=2
)
# search_resultの最初の要素を取得
first_result = search_result[0]

# valueとscoreを取り出す
value = first_result.payload.get('value')
score = first_result.score

print(f"Value: {value}")
print(f"Score: {score}")