#!/bin/bash

# =============================================================================
# 통합 RAG 시스템 간편 설치 스크립트
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=============================================================================="
echo -e "${GREEN}🚀 통합 RAG 시스템 간편 설치${NC}"
echo "=============================================================================="

# 현재 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 시스템 요구사항 확인
log_info "시스템 요구사항을 확인합니다..."

# Python 3 확인
if ! command -v python3 &> /dev/null; then
    log_error "Python 3가 설치되지 않았습니다."
    log_info "Python 설치: https://www.python.org/downloads/"
    exit 1
fi

# Docker 확인
if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다."
    log_info "Docker 설치: https://docs.docker.com/get-docker/"
    exit 1
fi

# Docker 데몬 상태 확인 및 시작
log_info "Docker 데몬 상태를 확인합니다..."
if ! docker info &> /dev/null; then
    log_warning "Docker 데몬이 실행되지 않고 있습니다."
    log_info "Docker Desktop을 시작합니다..."
    
    # macOS에서 Docker Desktop 시작
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/Applications/Docker.app" ]; then
            open -a Docker
            log_info "Docker Desktop 시작 중... 30초 대기합니다."
            sleep 30
            
            # Docker 데몬이 시작될 때까지 최대 60초 대기
            for i in {1..12}; do
                if docker info &> /dev/null; then
                    log_success "Docker 데몬이 시작되었습니다."
                    break
                fi
                log_info "Docker 데몬 시작 대기 중... ($i/12)"
                sleep 5
            done
            
            if ! docker info &> /dev/null; then
                log_error "Docker 데몬 시작에 실패했습니다."
                log_info "수동으로 Docker Desktop을 시작한 후 다시 실행해주세요."
                exit 1
            fi
        else
            log_error "Docker Desktop이 설치되지 않았습니다."
            log_info "Docker Desktop 설치: https://docs.docker.com/desktop/mac/install/"
            exit 1
        fi
    else
        log_error "Docker 데몬을 시작할 수 없습니다."
        log_info "수동으로 Docker를 시작한 후 다시 실행해주세요."
        exit 1
    fi
else
    log_success "Docker 데몬이 실행 중입니다."
fi

log_success "시스템 요구사항 확인 완료"

# 2. Elasticsearch 자동 시작
log_info "Elasticsearch를 시작합니다..."

# 기존 Elasticsearch 컨테이너 확인
if docker ps --format "table {{.Names}}" | grep -q elasticsearch; then
    log_success "Elasticsearch가 이미 실행 중입니다."
else
    log_info "Elasticsearch 컨테이너를 시작합니다..."
    
    # Docker Compose로 Elasticsearch 시작
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d elasticsearch
        
        # Elasticsearch가 준비될 때까지 대기
        log_info "Elasticsearch가 준비될 때까지 대기 중..."
        for i in {1..30}; do
            if curl -s http://localhost:9200 > /dev/null 2>&1; then
                log_success "Elasticsearch가 시작되었습니다."
                break
            fi
            log_info "Elasticsearch 시작 대기 중... ($i/30)"
            sleep 2
        done
        
        # 최종 확인
        if ! curl -s http://localhost:9200 > /dev/null 2>&1; then
            log_error "Elasticsearch 시작에 실패했습니다."
            log_info "수동으로 확인해주세요: docker-compose up -d elasticsearch"
            exit 1
        fi
        
        log_success "Elasticsearch가 정상적으로 시작되었습니다."
    else
        log_error "docker-compose.yml 파일을 찾을 수 없습니다."
        exit 1
    fi
fi

# 3. 가상환경 및 패키지 설치
log_info "Python 가상환경을 설정합니다..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    log_success "가상환경 생성 완료"
fi

source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1

log_info "패키지를 설치합니다..."
pip install -r requirements_unified.txt

log_success "패키지 설치 완료"

# 4. 환경 파일 설정
if [ ! -f ".env" ]; then
    log_info ".env 파일을 생성합니다..."
    cat > .env << 'EOF'
# Upstage API 키  
UPSTAGE_API_KEY=your_upstage_api_key

# Langsmith 설정 (선택사항)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=unified-rag-system
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Elasticsearch 설정
ELASTICSEARCH_URL=http://localhost:9200
INDEX_NAME=unified_rag

# PDF 디렉토리
PDF_DIR=pdf
EOF
    log_warning "API 키를 .env 파일에 설정해주세요!"
fi

# 5. 실행 권한 부여
chmod +x run_unified_rag.sh
chmod +x run_unified_rag_refactored.sh

log_success "설치 완료!"
echo
echo "=============================================================================="
echo -e "${GREEN}📝 다음 단계${NC}"
echo "=============================================================================="
echo "1. .env 파일에서 API 키 설정:"
echo "   - UPSTAGE_API_KEY=your_actual_key"
echo "2. 시스템 실행:"
echo "   ./run_unified_rag_refactored.sh  (권장 - 리팩토링된 버전)"
echo "   또는"
echo "   ./run_unified_rag.sh  (레거시 버전)"
echo "=============================================================================="
