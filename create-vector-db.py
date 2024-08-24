from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
import torch
import uuid


collection_name = "yasuda_ver.1.2"

# JSONLファイルの読み込みとベクトル化
data_path = "data/yasuda.jsonl"

# モデルとトクナイザーの読み込み
model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

qdrant_client = QdrantClient("http://localhost:6333")

qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=model.config.hidden_size, distance=Distance.COSINE
    ),
)

def encode_text(text: str):
    # トークンに分ける
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # ベクトル化する
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()


with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            # 各行を読み込んでJSONとしてパース
            data = json.loads(line.strip())  
            key = data["key"] # 質問
            value = data["value"] # 答え
            # IDを付与
            item_id = str(uuid.uuid4())
            # ベクトル化
            vector = encode_text(key)
            # payload(保存データを作成)
            payload = {"id": item_id, "key": key, "value": value}
            
            #登録
            qdrant_client.upload_collection(
                collection_name=collection_name,
                vectors=[vector],
                payload=[payload],
                ids=[item_id]
            )
            print(f"Inserted point ID: {item_id}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON line: {line}. Error: {str(e)}")
        except Exception as e:
            print(f"Failed to insert data: {str(e)}")

print("Data insertion completed.")