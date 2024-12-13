import torch
import os
import torch.nn as nn
from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# MongoDB URI 환경 변수에서 가져오기
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Atlas 연결 설정
client = MongoClient(MONGO_URI)
db = client["two_tower_model"]
user_embedding_collection = db["user_embeddings"]
product_embedding_collection = db["product_embeddings"]
train_dataset = db["train_dataset"]


# Autoencoder 모델 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),  # 512 -> 256
            nn.ReLU(),
            nn.Linear(256, 128),  # 256 -> 128
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(256, 512),  # 256 -> 512
        )

    def forward(self, x):
        return self.encoder(x)


# Autoencoder를 초기화하고 학습된 모델을 로드
autoencoder = Autoencoder()
autoencoder.eval()  # 학습된 모델 사용 시


# 학습된 모델 로드
def load_trained_model(model_path="product_model.pth"):
    """
    학습된 모델을 로드.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 256),  # 768: KoBERT 임베딩 차원
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드
    return model


# 유사도 계산 함수
def calculate_similarity(input_embedding, target_embeddings):
    """
    입력 임베딩과 대상 임베딩들 간의 cosine similarity를 계산.
    """
    similarities = cosine_similarity(input_embedding, target_embeddings).flatten()
    return similarities


def find_most_similar_anchor(userId, model):
    """
    사용자 임베딩을 기준으로 가장 유사한 anchor 상품을 반환.
    """
    # userId의 데이터 타입 확인 및 변환
    if isinstance(userId, str) and userId.isdigit():
        userId = int(userId)

    # 사용자 임베딩 가져오기
    user_data = user_embedding_collection.find_one({"userId": userId})

    if not user_data:
        raise ValueError(f"No embedding found for userId: {userId}")
    user_embedding = torch.tensor(
        user_data["embedding"][0], dtype=torch.float32
    ).unsqueeze(0)

    padding = torch.zeros((1, 768 - 512))
    user_embedding = torch.cat((user_embedding, padding), dim=1)

    # 사용자 임베딩 차원 축소 (768 -> 128)
    user_embedding = model[0](user_embedding)  # 첫 번째 레이어만 사용하여 차원 축소
    user_embedding = model[2](user_embedding)  # 마지막 레이어 적용 (128 차원)

    # Anchor 데이터 생성
    anchors, anchor_embeddings = [], []

    # Anchor 데이터를 product_model.pth에서 추출
    for _ in range(100):  # Anchor 데이터가 100개라고 가정
        random_input = torch.rand((1, 768))  # KoBERT 차원에 맞는 랜덤 데이터
        anchor_embedding = model(random_input).detach().numpy().flatten()
        anchors.append(f"Product_{len(anchors) + 1}")  # Anchor 상품 이름
        anchor_embeddings.append(anchor_embedding)

    anchor_embeddings = np.array(anchor_embeddings)

    print(f"User embedding dimension: {user_embedding.shape}")
    print(f"Anchor embedding dimension: {anchor_embeddings.shape}")

    # Cosine Similarity 계산
    similarities = calculate_similarity(
        user_embedding.detach().numpy().reshape(1, -1), anchor_embeddings
    )
    most_similar_index = np.argmax(similarities)

    return anchors[most_similar_index], anchor_embeddings[most_similar_index]


def find_most_similar_product(anchor_embedding, model):
    """
    Anchor 임베딩을 기반으로 학습된 positive/negative 상품 중 가장 유사한 상품을 반환.
    """
    train_embeddings, products = [], []
    # Anchor 데이터와 유사한 상품 임베딩을 생성
    for _ in range(100):  # 예시로 100개의 상품 임베딩을 계산한다고 가정
        random_input = torch.rand((1, 768))  # KoBERT 차원에 맞는 랜덤 데이터
        train_embedding = (
            model(random_input).detach().numpy().flatten()
        )  # 모델을 통해 임베딩 계산
        products.append(f"Product_{len(products) + 1}")  # 상품 이름
        train_embeddings.append(train_embedding)

    train_embeddings = np.array(train_embeddings)

    print(f"Anchor embedding dimension: {anchor_embedding.shape}")
    print(f"Train embedding dimension: {train_embeddings.shape}")

    # Cosine Similarity 계산
    similarities = calculate_similarity(
        anchor_embedding.reshape(1, -1), train_embeddings
    )
    most_similar_index = np.argmax(similarities)

    return products[most_similar_index], train_embeddings[most_similar_index]


def recommend_shop_product(similar_product_embedding):
    """
    학습된 상품과 쇼핑몰 상품 임베딩을 비교하여 최종 추천 상품 반환.
    """
    all_products = list(product_embedding_collection.find())
    shop_product_embeddings, shop_productIds = [], []

    for product in all_products:
        shop_productIds.append(product["productId"])
        shop_product_embeddings.append(product["embedding"])

    shop_product_embeddings = np.array(shop_product_embeddings)
    shop_product_embeddings = shop_product_embeddings.reshape(
        shop_product_embeddings.shape[0], -1
    )

    # Shop 제품 임베딩을 NumPy 배열로 변환
    shop_product_embeddings = np.array(shop_product_embeddings)

    # Autoencoder로 차원 축소 (512 -> 128)
    shop_product_embeddings_reduced = (
        autoencoder.encoder(torch.tensor(shop_product_embeddings).float())
        .detach()
        .numpy()
    )

    # similar_product_embedding을 (1, 128)로 변환
    similar_product_embedding = similar_product_embedding.reshape(1, -1)

    print(f"Similar product embedding dimension: {similar_product_embedding.shape}")
    print(f"Shop product embedding dimension: {shop_product_embeddings_reduced.shape}")

    # Cosine Similarity 계산
    similarities = calculate_similarity(
        similar_product_embedding, shop_product_embeddings_reduced
    )

    # 상위 3개 상품 인덱스 추출
    top_3_indices = np.argsort(similarities)[-3:][
        ::-1
    ]  # 유사도가 높은 상위 3개 (내림차순)

    # 상위 3개 상품 ID 반환
    top_3_productIds = [shop_productIds[i] for i in top_3_indices]

    return top_3_productIds
