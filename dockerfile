# 기본 이미지 선택
FROM python:3.11-slim

# 작업 디렉터리 설정
WORKDIR /app

# 로컬 파일들을 도커 컨테이너의 작업 디렉터리로 복사
COPY . /app

# pip 최신 버전으로 업데이트
RUN python -m pip install --upgrade pip

# 필요한 Python 패키지를 설치하기 위해 requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# 파이썬 스크립트 실행
CMD ["python", "app.py"]
