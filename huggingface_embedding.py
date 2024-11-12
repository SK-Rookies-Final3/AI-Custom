from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
import torch
from torch.nn import Embedding

# MongoDB Atlas 연결 설정
client = MongoClient(
    "mongodb+srv://waseoke:rookies3@cluster0.ps7gq.mongodb.net/test?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true"
)
db = client["패션"]
product_collection = db["여성의류"]

# Hugging Face의 한국어 BERT 모델 및 토크나이저 로드 (예: klue/bert-base)
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = BertModel.from_pretrained("klue/bert-base")


# 상품 타워: 데이터 임베딩
def embed_product_data(product_data):
    # 상품명과 상세 정보 임베딩 (BERT)
    text = product_data.get("title", "") + " " + product_data.get("description", "")
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    outputs = model(**inputs)
    text_embedding = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링으로 벡터화

    # 카테고리 및 색상 정보 임베딩 (임베딩 레이어)
    category_embedding_layer = Embedding(num_embeddings=50, embedding_dim=16)
    color_embedding_layer = Embedding(num_embeddings=20, embedding_dim=8)

    category_id = product_data.get("category_id", 0)  # 카테고리 ID, 기본값 0
    color_id = product_data.get("color_id", 0)  # 색상 ID, 기본값 0

    category_embedding = category_embedding_layer(torch.tensor([category_id]))
    color_embedding = color_embedding_layer(torch.tensor([color_id]))

    # 최종 임베딩 벡터 결합
    product_embedding = torch.cat(
        (text_embedding, category_embedding, color_embedding), dim=1
    )
    return product_embedding.detach().numpy()


# MongoDB에서 데이터 가져오기
product_data = product_collection.find_one({"product_id": "1"})  # 특정 상품 ID

# 임베딩 수행
if product_data:
    product_embedding = embed_product_data(product_data)
    print("Product Embedding:", product_embedding)
else:
    print("Product not found.")
