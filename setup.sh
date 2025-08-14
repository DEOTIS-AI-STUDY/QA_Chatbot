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

log_success "시스템 요구사항 확인 완료"

# 2. 가상환경 및 패키지 설치
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

# 3. 환경 파일 설정
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

# 4. 실행 권한 부여
chmod +x run_unified_rag.sh

log_success "설치 완료!"
echo
echo "=============================================================================="
echo -e "${GREEN}📝 다음 단계${NC}"
echo "=============================================================================="
echo "1. .env 파일에서 API 키 설정:"
echo "   - UPSTAGE_API_KEY=your_actual_key"
echo "2. 시스템 실행:"
echo "   ./run_unified_rag.sh"
echo "=============================================================================="
