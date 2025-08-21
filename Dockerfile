# FastAPI RAG 시스템용 Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 디렉토리 생성
RUN mkdir -p /app/src/core /app/src/utils /app/src/api

# 환경 설정 파일 복사 (프로젝트 루트의 .env.prod만 사용)
COPY .env.prod ./

# 상위 디렉토리의 core, utils 모듈 복사
COPY src/core/ ./src/core/
COPY src/utils/ ./src/utils/

# API 애플리케이션 코드 복사
COPY src/api/ ./src/api/

# 포트 노출
EXPOSE 8110

# 환경 변수 설정
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8110/health || exit 1

# 서버 실행 (main.py는 src/api 디렉토리 안에 있음)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8110"]
