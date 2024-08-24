import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

load_dotenv()  # .env ファイルから環境変数を読み込む

app = FastAPI()

# モデルとトークナイザーの読み込み
model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

qdrant_client = QdrantClient("http://localhost:6333")

collection_name = "yasuda_ver.1.2"

class Item(BaseModel):
    description: str

class ItemUpdate(BaseModel):
    id: str
    description: str

class Query(BaseModel):
    query: str

line_channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
line_channel_secret = os.getenv('LINE_CHANNEL_SECRET')
api_key = os.getenv("API_KEY")

if not all([line_channel_access_token, line_channel_secret]):
    raise ValueError("必要な環境変数が設定されていません。")

line_bot_api = LineBotApi(line_channel_access_token)
handler = WebhookHandler(line_channel_secret)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

def encode_text(text: str):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

def get_response(text: str):
    # ユーザーの入力をベクトル化
    query_vector = encode_text(text)
    # ユーザーの入力したベクトルとDB内のベクトルのコサイン類似度を計算し
    # 一番類似度の高いものを得る
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1
    )
    # search_resultの最初の要素を取得
    first_result = search_result[0]
    # valueとscoreを取り出す
    value = first_result.payload.get('value')
    score = first_result.score
    return value, score

@app.get("/")
def get_api_info():
    return {"yasuda-bot-api": "ver.1.0"}

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(status_code=400, detail="X-Line-Signature header not found")

    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return JSONResponse(content="OK")

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    asyncio.create_task(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    try:
        # ユーザーのメッセージを取得
        user_message = event.message.text
        # Qdrantから一番類似度の高い内容を取得
        value, score = get_response(user_message)
        # レスポンスの作成
        response = f'{value}({score})'
        # 類似度が低いものに関しては
        if score < 0.6:
            response = f'そのご質問に関するデータがないためお答えすることができません。({score})'
        # LINE APIでレスポンスを送信
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
        
    except Exception as e:
        print(f"Error handling message: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)