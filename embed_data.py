from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
from flask import Blueprint, jsonify
import torch
import os
from torch.nn import Embedding
from dotenv import load_dotenv

product_bp = Blueprint("product", __name__)
user_bp = Blueprint("user", __name__)

load_dotenv()

# MongoDB URI 환경 변수에서 가져오기
MONGO_URI = os.getenv("MONGO_URI")

MONGO_URI_HONG = os.getenv("MONGO_URI_HONG")

# MongoDB Atlas 연결 설정
client = MongoClient(MONGO_URI)
db = client["two_tower_model"]
product_collection = db["product_tower"]
user_collection = db["user_tower"]
product_embedding_collection = db["product_embeddings"]  # 상품 임베딩 저장
user_embedding_collection = db["user_embeddings"]  # 사용자 임베딩 저장

client_HONG = MongoClient(MONGO_URI_HONG)
product_embedding_prev = client_HONG["product_embedding_prev"]
product_data = product_embedding_prev["product_data"]
user_actions = client_HONG["user_actions"]
user_purchases = user_actions["user_purchases"]

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
    """
    상품 데이터를 임베딩하는 함수. 데이터셋의 여러 필드에서 정보를 추출해 벡터화.
    """
    # 텍스트 기반 필드 임베딩
    # `tsf_context_dist_vector` 필드에 저장된 텍스트 데이터를 KoBERT를 사용해 임베딩 -> 768차원
    context_text = product_data["data"]["tsf_context_dist_vector"][0]
    inputs = tokenizer(
        context_text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    outputs = model(**inputs)
    text_embedding = outputs.last_hidden_state.mean(dim=1)  # BERT 임베딩

    # `category`, `fiber_composition`, `color`, `category_specification` 같은 필드는 명목형 데이터이므로, 고유 ID를 임베딩 레이어에 입력하여 벡터화
    category_embedding_layer = Embedding(num_embeddings=100, embedding_dim=16)
    fiber_composition_embedding_layer = Embedding(num_embeddings=50, embedding_dim=8)
    color_embedding_layer = Embedding(num_embeddings=20, embedding_dim=8)
    category_specification_embedding_layer = Embedding(
        num_embeddings=30, embedding_dim=8
    )

    # 각 속성값에 대해 고유 ID를 생성하고, 임베딩 레이어 입력값으로 사용
    category_id = hash(product_data["data"]["clothes"]["category"][0]) % 100
    fiber_composition_id = (
        hash(product_data["data"]["clothes"]["fiber_composition"][0]) % 50
    )
    color_id = hash(product_data["data"]["clothes"]["color"][0]) % 20
    category_specification_id = (
        hash(
            product_data["data"]["reinforced_feature_value"]["category_specification"][
                0
            ]
        )
        % 30
    )

    # 각 속성에 대한 임베딩 생성
    category_embedding = category_embedding_layer(torch.tensor([category_id]))
    fiber_composition_embedding = fiber_composition_embedding_layer(
        torch.tensor([fiber_composition_id])
    )
    color_embedding = color_embedding_layer(torch.tensor([color_id]))
    category_specification_embedding = category_specification_embedding_layer(
        torch.tensor([category_specification_id])
    )

    # 숫자형 데이터는 그대로 텐서로 변환
    metadata_vector = product_data["data"]["tsf_clothes_metadata_vector_concator"][0]
    metadata_vector = torch.tensor(metadata_vector, dtype=torch.float32)

    # 모든 임베딩 결합
    combined_embedding = torch.cat(
        [
            text_embedding,
            category_embedding.view(1, -1),
            fiber_composition_embedding.view(1, -1),
            color_embedding.view(1, -1),
            category_specification_embedding.view(1, -1),
            metadata_vector.view(1, -1),
        ],
        dim=1,
    )

    # 결합된 벡터를 평균 풀링을 사용해 512차원으로 맞춤
    product_embedding = torch.nn.functional.adaptive_avg_pool1d(
        combined_embedding.unsqueeze(0), 512
    ).squeeze(0)

    return product_embedding.detach().numpy()


# 사용자 타워: 데이터 임베딩
def embed_user_data(user_data):
    """
    사용자 데이터를 임베딩하는 함수. 나이, 성별, 신체 정보, 주문 상품 데이터 등을 포함.
    """
    embedding_layer = Embedding(num_embeddings=100, embedding_dim=128)

    # `data` 키 내부에서 필요한 값을 추출
    user_details = user_data.get("data", {})

    # 성별 임베딩 ('M'은 0, 'F'는 1로 인코딩)
    gender = user_details.get("gender", "M")
    gender_id = 0 if gender == "M" else 1

    # 스케일링된 키와 몸무게 계산
    height = user_details.get("height", 150)  # 기본값 150
    weight = user_details.get("weight", 50)  # 기본값 50

    if not (min_height <= height <= max_height):
        raise ValueError(
            f"Invalid height: {height}. Expected range: {min_height}-{max_height}"
        )
    if not (min_weight <= weight <= max_weight):
        raise ValueError(
            f"Invalid weight: {weight}. Expected range: {min_weight}-{max_weight}"
        )

    scaled_height = (height - min_height) * 99 // (max_height - min_height)
    scaled_weight = (weight - min_weight) * 99 // (max_weight - min_weight)

    # 개별 값에 대해 임베딩 생성
    age_embedding = embedding_layer(torch.tensor([user_details.get("age", 0)])).view(
        1, -1
    )
    gender_embedding = embedding_layer(torch.tensor([gender_id])).view(1, -1)
    height_embedding = embedding_layer(torch.tensor([scaled_height])).view(1, -1)
    weight_embedding = embedding_layer(torch.tensor([scaled_weight])).view(1, -1)

    # 주문한 상품의 임베딩 벡터 가져오기
    order_product_ids = user_details.get("productIds", [])
    product_embeddings = []

    for product_id in order_product_ids:
        product_data = product_embedding_collection.find_one({"productId": product_id})
        if not product_data or "embedding" not in product_data:
            print(f"Product ID {product_id}의 임베딩 벡터를 찾을 수 없습니다.")
            continue
        product_embeddings.append(
            torch.tensor(product_data["embedding"], dtype=torch.float32)
        )

    # 여러 상품의 임베딩 벡터 평균 계산
    if product_embeddings:
        product_embeddings = torch.stack(product_embeddings).mean(dim=0).view(1, -1)
    else:
        product_embeddings = torch.zeros((1, 512))  # 임베딩이 없는 경우 기본값

    # 모든 임베딩 벡터 결합
    combined_embedding = torch.cat(
        [
            age_embedding,
            gender_embedding,
            height_embedding,
            weight_embedding,
            product_embeddings,
        ],
        dim=1,
    )

    # 최종 임베딩 벡터 차원 조정 (512차원)
    user_embedding = torch.nn.functional.adaptive_avg_pool1d(
        combined_embedding.unsqueeze(0), 512
    ).squeeze(0)

    return user_embedding.detach().numpy()


# API 호출해서 상품 데이터 가져오고 임베딩하여 저장
@product_bp.route("/ai-api/products/<int:productId>", methods=["GET"])
def get_products_data_embedding(productId):
    try:
        all_product_data = product_data.find({"productId": productId})

        for product_datas in all_product_data:
            product_embedding = embed_product_data(product_datas)
            print(
                f"Product ID {product_datas['productId']} Embedding: {product_embedding}"
            )

            # MongoDB Atlas의 product_embeddings 컬렉션에 임베딩 저장
            product_embedding_collection.update_one(
                {"productId": product_datas["productId"]},  # productId 기준으로 찾기
                {
                    "$set": {"embedding": product_embedding.tolist()}
                },  # 벡터를 리스트 형태로 저장
                upsert=True,  # 기존 항목이 없으면 새로 삽입
            )
            print(
                f"Embedding saved to MongoDB Atlas for Product ID {product_datas['productId']}."
            )

        return (
            jsonify(
                {
                    "message": "Product data saved successfully",
                    "productId": productId,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"message": str(e)}), 500


# API 호출해서 사용자 데이터 가져오고 임베딩하여 저장
@user_bp.route("/ai-api/users/<int:userId>", methods=["GET"])
def get_users_data_embedding(userId):
    try:
        if not isinstance(userId, int):
            return jsonify({"message": "Invalid userId format"}), 400

        all_user_data = user_purchases.find({"userId": userId})

        for user_datas in all_user_data:
            try:
                user_embedding = embed_user_data(user_datas)
                print(f"User ID {user_datas['userId']} Embedding:", user_embedding)

                # MongoDB에 저장
                user_embedding_collection.update_one(
                    {"userId": user_datas["userId"]},
                    {"$set": {"embedding": user_embedding.tolist()}},
                    upsert=True,
                )
                print(f"Embedding saved to MongoDB for userId {user_datas['userId']}.")
            except KeyError as e:
                print(f"KeyError: {e} in user data {user_datas}")
            except Exception as e:
                print(f"Error embedding user data: {e}")

        return (
            jsonify(
                {
                    "message": "User data saved successfully",
                    "userId": userId,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"message": str(e)}), 500
