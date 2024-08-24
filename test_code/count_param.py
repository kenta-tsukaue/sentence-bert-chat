from transformers import AutoTokenizer, AutoModel

model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# パラメータの総数を計算
total_params = sum(p.numel() for p in model.parameters())

print(f"モデルの総パラメータ数: {total_params}")