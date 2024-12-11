import torch
import torch.nn.functional as F
import os
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv

load_dotenv()

# MongoDB URI 환경 변수에서 가져오기
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Atlas 연결 설정
client = MongoClient(MONGO_URI)
db = client["two_tower_model"]
train_dataset = db["train_dataset"]

# KoBERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertModel.from_pretrained("monologg/kobert")


# 상품 임베딩 함수
def embed_product_data(clothes):
    """
    상품 데이터를 KoBERT로 임베딩
    """
    # 'clothes' 정보에서 텍스트로 결합할 수 있는 키들 추출
    clothes_info = []

    # 카테고리, 소재, 색상, 브랜드 등을 텍스트로 결합
    if "category" in clothes:
        clothes_info.append(" ".join(clothes["category"]))
    if "material" in clothes:
        clothes_info.append(" ".join(clothes["material"]))
    if "color" in clothes:
        clothes_info.append(" ".join(clothes["color"]))
    if "brand" in clothes:
        clothes_info.append(" ".join(clothes["brand"]))
    if "specific_context" in clothes:
        clothes_info.append(" ".join(clothes["specific_context"]))

    # 결합된 텍스트
    text = " ".join(clothes_info)

    # 텍스트를 토크나이저에 넣어 임베딩 생성
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding


# PyTorch Dataset 정의
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        anchor = torch.tensor(data["anchor_embedding"], dtype=torch.float32)
        positive = torch.tensor(data["positive_embedding"], dtype=torch.float32)
        negative = torch.tensor(data["negative_embedding"], dtype=torch.float32)
        return anchor, positive, negative


# MongoDB에서 데이터셋 로드 및 임베딩 변환
def prepare_training_data(verbose=False):
    dataset = list(train_dataset.find())
    if not dataset:
        raise ValueError("No training data found in MongoDB.")

    # Anchor, Positive, Negative 임베딩 생성
    embedded_dataset = []
    for idx, entry in enumerate(dataset):
        try:
            # 'product' 키가 존재하는지 확인
            if (
                "anchor" not in entry
                or "positive" not in entry
                or "negative" not in entry
            ):
                print(f"Missing anchor/positive/negative data in sample {idx + 1}")
                continue  # 필요한 데이터가 없으면 건너뛰기

            # Anchor, Positive, Negative 데이터 임베딩
            anchor_embedding = embed_product_data(entry["anchor"]["clothes"])
            positive_embedding = embed_product_data(entry["positive"]["clothes"])
            negative_embedding = embed_product_data(entry["negative"]["clothes"])

            if (
                anchor_embedding is None
                or positive_embedding is None
                or negative_embedding is None
            ):
                print(f"Skipping sample {idx + 1} due to missing embeddings.")
                continue  # 임베딩이 정상적으로 생성되지 않으면 건너뛰기

            # 임베딩 결과 저장
            embedded_dataset.append(
                {
                    "anchor_embedding": anchor_embedding,
                    "positive_embedding": positive_embedding,
                    "negative_embedding": negative_embedding,
                }
            )
        except Exception as e:
            print(f"Error embedding data at sample {idx + 1}: {e}")

    return TripletDataset(embedded_dataset)


# 데이터셋 검증용 함수
def validate_embeddings():
    """
    데이터셋 임베딩을 생성하고 각 임베딩의 일부를 출력하여 확인.
    """
    print("Validating embeddings...")
    triplet_dataset = prepare_training_data(verbose=True)
    print(f"Total samples: {len(triplet_dataset)}")
    return triplet_dataset


# Triplet Loss를 학습시키는 함수
def train_triplet_model(
    product_model, train_loader, num_epochs=30, learning_rate=0.0001, margin=0.2
):
    optimizer = Adam(product_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        product_model.train()
        total_loss = 0

        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()

            # Forward pass
            anchor_vec = product_model(anchor)
            positive_vec = product_model(positive)
            negative_vec = product_model(negative)

            # Triplet loss 계산
            positive_distance = F.pairwise_distance(anchor_vec, positive_vec)
            negative_distance = F.pairwise_distance(anchor_vec, negative_vec)
            triplet_loss = torch.clamp(
                positive_distance - negative_distance + margin, min=0
            ).mean()

            # 역전파와 최적화
            triplet_loss.backward()
            optimizer.step()

            total_loss += triplet_loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
        )

    return product_model


# 모델 학습 파이프라인
def main():
    # 모델 초기화 (예시 모델)
    product_model = torch.nn.Sequential(
        torch.nn.Linear(768, 256),  # 768: KoBERT 임베딩 차원
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )

    # 데이터 준비
    triplet_dataset = prepare_training_data()
    train_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

    # 모델 학습
    trained_model = train_triplet_model(product_model, train_loader)

    # 학습된 모델 저장
    torch.save(trained_model.state_dict(), "product_model.pth")
    print("Model training completed and saved.")
    print(validate_embeddings())


if __name__ == "__main__":
    main()
