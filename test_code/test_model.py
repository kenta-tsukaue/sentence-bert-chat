from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


load_dotenv()  # .env ファイルから環境変数を読み込む

# モデルとトークナイザーの読み込み
model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def encode_text(text: str):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

def tokenize_text(text: str):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    print(tokens)

if __name__ == "__main__":
    text_1 = "大学の所在地は名古屋です"
    text_2 = "大学は名護屋にあります"
    vec_1 = encode_text(text_1)
    vec_2 = encode_text(text_2)

    # コサイン類似度を計算
    cosine_similarity = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    print(cosine_similarity)