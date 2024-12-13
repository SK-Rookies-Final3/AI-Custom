from flask import Flask, jsonify
from flask_cors import CORS
from embed_data import product_bp, user_bp
import os
from dotenv import load_dotenv
from pymongo import MongoClient

app = Flask(__name__)

# CORS Allowed All Origins
CORS(app)


## DB Creation for Preference Tracker
load_dotenv()

# MongoDB URI 환경 변수에서 가져오기
MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI_HONG = os.getenv("MONGO_URI_HONG")

client = MongoClient(MONGO_URI_HONG)

db = client["user_preference_list"]  # 데이터베이스 선택
user_preference_collection = db["user_preference"]  # 컬렉션 선택

## router ##
# embed_data
app.register_blueprint(product_bp)
app.register_blueprint(user_bp)


## EOL router ##


from subprocess import run
from calculate_cosine_similarity import (
    load_trained_model,
    find_most_similar_anchor,
    find_most_similar_product,
    recommend_shop_product,
)


def execute_script(script_name):
    """
    Helper function to execute a Python script.
    """
    print(f"Executing {script_name}...")
    result = run(["python", script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script_name} executed successfully.")
    else:
        print(f"Error executing {script_name}:")
        print(result.stderr)


@app.route("/infer-api/product/preference/<int:userId>", methods=["GET"])
def main(userId):
    # # Step 0: 모델 학습
    # print("Step 0: 모델 학습 중...")
    # execute_script("train_model.py")

    # Step 1: 쇼핑물 상품과 사용자 임베딩 -> 상품, 사용자 정보 가져오는 API 엔드포인트
    # print("쇼핑물 상품과 사용자 임베딩...")
    # execute_script("embed_data.py")

    # Step 2: product_model.pth 불러오기
    print("product_model.pth 불러오는 중...")
    model = load_trained_model("product_model.pth")

    # Step 3: 추천을 위한 사용자 ID 입력
    print(f"사용자 ID: {userId}에게 추천해줄 상품 찾는 중...")

    try:
        # Step 4: 사용자와 가장 유사한 anchor 찾기
        print(f"사용자 ID: {userId} 와 가장 유사한 anchor 찾는 중...")
        most_similar_anchor, most_similar_anchor_embedding = find_most_similar_anchor(
            userId, model
        )
        print(f"가장 유사한 anchor: {most_similar_anchor}")

        # Step 5: anchor와 가장 유사한 상품 찾기
        print("anchor와 가장 유사한 학습 상품 찾는 중...")
        most_similar_product, most_similar_product_embedding = (
            find_most_similar_product(most_similar_anchor_embedding, model)
        )
        print(f"anchor와 가장 유사한 학습 상품 ID: {most_similar_product}")

        # Step 6: 쇼핑몰 상품 추천
        print("추천 쇼핑몰 상품 찾는 중...")
        recommended_productId = recommend_shop_product(most_similar_product_embedding)
        print(f"추천 쇼핑몰 상품 ID: {recommended_productId}")

        # save MongoDB to repres.data
        user_preference_data = {
            "userId": userId,
            "recommended_productId": recommended_productId,
        }
        user_preference_collection.insert_one(user_preference_data)

        
        
        return jsonify(
            {
                "user_id": userId,
                "recommended_productId": recommended_productId,
            },
            200,
        )

    except Exception as e:
        print(f"An error occurred during the recommendation process: {e}")


if __name__ == "__main__":
    app.run(
        os.getenv("CUSTOM_RUN_HOST"), port=int(os.getenv("CUSTOM_RUN_PORT")), debug=True
    )
