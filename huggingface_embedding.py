from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
import torch
from torch.nn import Embedding

# MongoDB Atlas 연결 설정
client = MongoClient("mongodb+srv://waseoke:rookies3@cluster0.ps7gq.mongodb.net/test?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true")
db = client["two_tower_model"]
product_collection = db["product_tower"]
user_collection = db['user_tower']
product_embedding_collection = db["product_embeddings"]  # 상품 임베딩을 저장할 컬렉션
user_embedding_collection = db["user_embeddings"]  # 사용자 임베딩을 저장할 컬렉션

# Hugging Face의 한국어 BERT 모델 및 토크나이저 로드 (예: klue/bert-base)
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = BertModel.from_pretrained("klue/bert-base")

# Height와 Weight 스케일링에 필요한 값 설정
min_height = 50
max_height = 250
min_weight = 30
max_weight = 200

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

# 사용자 타워: 데이터 임베딩
def embed_user_data(user_data):
    # 나이, 성별, 키, 몸무게 임베딩 (임베딩 레이어)
    embedding_layer = Embedding(num_embeddings=100, embedding_dim=32)  # 임의로 설정된 예시 값

    # 예를 들어 성별을 'M'은 0, 'F'는 1로 인코딩
    gender_id = 0 if user_data['gender'] == 'M' else 1

    # 스케일링 적용
    height = user_data['height']
    weight = user_data['weight']

    if not (min_height <= height <= max_height):
        raise ValueError(f"Invalid height value: {height}. Expected range: {min_height}-{max_height}")
    if not (min_weight <= weight <= max_weight):
        raise ValueError(f"Invalid weight value: {weight}. Expected range: {min_weight}-{max_weight}")

    scaled_height = (height - min_height) * 99 // (max_height - min_height)
    scaled_weight = (weight - min_weight) * 99 // (max_weight - min_weight)
    
    age_embedding = embedding_layer(torch.tensor([user_data['age']]))
    gender_embedding = embedding_layer(torch.tensor([gender_id]))
    height_embedding = embedding_layer(torch.tensor([scaled_height]))
    weight_embedding = embedding_layer(torch.tensor([scaled_weight]))

    # 최종 임베딩 벡터 결합
    user_embedding = torch.cat((age_embedding, gender_embedding, height_embedding, weight_embedding), dim=1)
    return user_embedding.detach().numpy()

# MongoDB Atlas에서 데이터 가져오기
all_products = product_collection.find() # 모든 상품 데이터 가져오기
all_users = user_collection.find()  # 모든 사용자 데이터 가져오기

# 상품 임베딩 수행
for product_data in all_products:
    product_embedding = embed_product_data(product_data)
    print(f"Product ID {product_data['product_id']} Embedding: {product_embedding}")

    # MongoDB Atlas의 product_embeddings 컬렉션에 임베딩 저장
    product_embedding_collection.update_one(
        {"product_id": product_data["product_id"]},  # product_id 기준으로 찾기
        {"$set": {"embedding": product_embedding.tolist()}},  # 벡터를 리스트 형태로 저장
        upsert=True  # 기존 항목이 없으면 새로 삽입
    )
    print(f"Embedding saved to MongoDB Atlas for Product ID {product_data['product_id']}.")

# 사용자 임베딩 수행
for user_data in all_users:
    try:
        user_embedding = embed_user_data(user_data)
        print(f"User ID {user_data['user_id']} Embedding:", user_embedding)

        # MongoDB Atlas의 user_embeddings 컬렉션에 임베딩 저장
        user_embedding_collection.update_one(
            {"user_id": user_data["user_id"]},  # user_id 기준으로 찾기
            {"$set": {"embedding": user_embedding.tolist()}},  # 벡터를 리스트 형태로 저장
            upsert=True  # 기존 항목이 없으면 새로 삽입
        )
        print(f"Embedding saved to MongoDB Atlas for user_id {user_data['user_id']}.")
    except ValueError as e:
        print(f"Skipping user_id {user_data['user_id']} due to error: {e}")