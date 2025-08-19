#!/bin/bash

# FastAPI RAG 시스템 실행 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 FastAPI RAG 시스템 실행 스크립트${NC}"
echo "=================================="

# 현재 디렉토리 확인
if [[ ! -f "main.py" ]]; then
    echo -e "${RED}❌ main.py 파일을 찾을 수 없습니다. src/api 디렉토리에서 실행하세요.${NC}"
    exit 1
fi

# Python 가상환경 확인
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️ 가상환경이 활성화되지 않았습니다.${NC}"
    echo "다음 명령어로 가상환경을 활성화하세요:"
    echo "source ../../.venv/bin/activate"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 의존성 설치 확인
echo -e "${BLUE}📦 의존성 확인 중...${NC}"
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}📦 FastAPI 의존성을 설치합니다...${NC}"
    pip install -r requirements.txt
fi

# Elasticsearch 연결 확인
echo -e "${BLUE}🔍 Elasticsearch 연결 확인 중...${NC}"
if ! curl -s http://localhost:9200/_health > /dev/null; then
    echo -e "${YELLOW}⚠️ Elasticsearch 서버에 연결할 수 없습니다.${NC}"
    echo "Elasticsearch를 시작하세요:"
    echo "  docker-compose up -d elasticsearch"
    echo "  또는: ../../setup.sh"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Ollama 연결 확인 (선택적)
echo -e "${BLUE}🦙 Ollama 연결 확인 중...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${YELLOW}⚠️ Ollama 서버에 연결할 수 없습니다.${NC}"
    echo "Ollama를 시작하거나 Upstage API를 사용하세요."
fi

# 서버 설정
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
RELOAD=${RELOAD:-false}

# 서버 실행
echo -e "${GREEN}🚀 FastAPI 서버를 시작합니다...${NC}"
echo "  📍 주소: http://${HOST}:${PORT}"
echo "  📚 API 문서: http://${HOST}:${PORT}/docs"
echo "  🌐 웹 인터페이스: file://$(pwd)/web.html"
echo "  ⏹️  중지: Ctrl+C"
echo

if [[ "$RELOAD" == "true" ]]; then
    echo -e "${BLUE}🔄 개발 모드 (자동 리로드)${NC}"
    uvicorn main:app --host $HOST --port $PORT --reload
else
    echo -e "${BLUE}🏭 프로덕션 모드${NC}"
    uvicorn main:app --host $HOST --port $PORT
fi
