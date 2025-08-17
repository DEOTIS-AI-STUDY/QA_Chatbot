#!/bin/bash

# =============================================================================
# 통합 RAG 시스템 실행 스크립트
# =============================================================================

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "통합 RAG 시스템을 시작합니다..."
log_info "작업 디렉토리: $SCRIPT_DIR"

# =============================================================================
# 1. 환경 확인
# =============================================================================

log_info "시스템 환경을 확인합니다..."

# Python 확인
if ! command -v python3 &> /dev/null; then
    log_error "Python 3가 설치되지 않았습니다."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
log_success "Python 확인: $PYTHON_VERSION"

# Docker 확인
if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다."
    log_info "Docker 설치: https://docs.docker.com/get-docker/"
    exit 1
fi

# Docker 데몬 상태 확인 및 자동 시작
if ! docker info &> /dev/null; then
    log_warning "Docker 데몬이 실행되지 않고 있습니다."
    
    # macOS에서 Docker Desktop 자동 시작
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/Applications/Docker.app" ]; then
            log_info "Docker Desktop을 시작합니다..."
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
fi

log_success "Docker 확인: $(docker --version)"

# =============================================================================
# 2. 가상환경 설정
# =============================================================================

log_info "Python 가상환경을 설정합니다..."

# 가상환경 생성 (없는 경우)
if [ ! -d ".venv" ]; then
    log_info "가상환경을 생성합니다..."
    python3 -m venv .venv
    log_success "가상환경 생성 완료"
else
    log_success "기존 가상환경을 사용합니다"
fi

# 가상환경 활성화
source .venv/bin/activate
log_success "가상환경 활성화 완료"

# pip 업그레이드
log_info "pip를 업그레이드합니다..."
pip install --upgrade pip > /dev/null 2>&1

# =============================================================================
# 3. 패키지 설치
# =============================================================================

log_info "필요한 패키지를 설치합니다..."

# requirements_unified.txt 파일 확인
if [ -f "requirements_unified.txt" ]; then
    log_info "requirements_unified.txt에서 패키지 설치 중..."
    pip install -r requirements_unified.txt > /dev/null 2>&1
    log_success "requirements_unified.txt 패키지 설치 완료"
else
    log_warning "requirements_unified.txt 파일이 없습니다. 개별 패키지를 설치합니다..."
    
    # 핵심 패키지들 설치
    log_info "핵심 패키지 설치 중..."
    pip install streamlit elasticsearch==8.11.0 pydantic==2.11.7 > /dev/null 2>&1

    # LangChain 패키지들 설치
    log_info "LangChain 패키지 설치 중..."
    pip install langchain langchain-core langchain-community langchain-text-splitters > /dev/null 2>&1

    # LLM 패키지들 설치
    log_info "LLM 패키지 설치 중..."
    pip install langchain-upstage==0.7.1 langchain-ollama > /dev/null 2>&1

    # 임베딩 및 기타 패키지들 설치
    log_info "임베딩 및 기타 패키지 설치 중..."
    pip install langchain-huggingface sentence-transformers torch transformers > /dev/null 2>&1
    pip install PyPDF2 psutil python-dotenv urllib3 > /dev/null 2>&1

    # Langsmith (선택사항)
    log_info "Langsmith 패키지 설치 중..."
    pip install langsmith > /dev/null 2>&1
    
    log_success "개별 패키지 설치 완료"
fi

# =============================================================================
# 4. Elasticsearch 설정
# =============================================================================

log_info "Elasticsearch를 설정합니다..."

# Docker Compose 파일 확인
if [ ! -f "docker-compose.yml" ]; then
    log_warning "docker-compose.yml 파일이 없습니다. 생성합니다..."
    
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch-rag
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - elastic

networks:
  elastic:
    driver: bridge

volumes:
  elasticsearch_data:
    driver: local
EOF
    
    log_success "docker-compose.yml 파일 생성 완료"
fi

# Elasticsearch 컨테이너 확인 및 시작
if ! docker ps | grep -q elasticsearch-rag; then
    log_info "Elasticsearch 컨테이너를 시작합니다..."
    docker-compose up -d elasticsearch
    
    # Elasticsearch 준비 대기
    log_info "Elasticsearch가 준비될 때까지 대기합니다..."
    for i in {1..30}; do
        if curl -s http://localhost:9200 > /dev/null 2>&1; then
            log_success "Elasticsearch 준비 완료"
            break
        fi
        
        if [ $i -eq 30 ]; then
            log_error "Elasticsearch 시작 실패 (30초 타임아웃)"
            exit 1
        fi
        
        echo -n "."
        sleep 1
    done
    echo
else
    log_success "Elasticsearch가 이미 실행 중입니다"
fi

# Elasticsearch 연결 테스트
if curl -s http://localhost:9200 > /dev/null; then
    ES_VERSION=$(curl -s http://localhost:9200 | python3 -c "import sys, json; print(json.load(sys.stdin)['version']['number'])" 2>/dev/null || echo "Unknown")
    log_success "Elasticsearch 연결 성공 (v$ES_VERSION)"
else
    log_error "Elasticsearch 연결 실패"
    exit 1
fi

# =============================================================================
# 5. 디렉토리 생성
# =============================================================================

log_info "필요한 디렉토리를 생성합니다..."

# PDF 디렉토리 생성
mkdir -p pdf
mkdir -p data
mkdir -p logs

log_success "디렉토리 생성 완료"

# =============================================================================
# 6. 환경변수 확인
# =============================================================================

log_info "환경변수를 확인합니다..."

# .env 파일 확인
if [ -f ".env" ]; then
    log_success ".env 파일이 존재합니다"
    
    # API 키 확인
    source .env 2>/dev/null || true
    
    if [ -n "$UPSTAGE_API_KEY" ] && [ "$UPSTAGE_API_KEY" != "your_upstage_api_key" ]; then
        log_success "Upstage API 키 설정됨"
    else
        log_warning "Upstage API 키가 설정되지 않았습니다"
    fi
    
    if [ -n "$LANGSMITH_API_KEY" ]; then
        log_success "Langsmith API 키 설정됨"
    else
        log_warning "Langsmith API 키가 설정되지 않았습니다"
    fi
else
    log_warning ".env 파일이 없습니다. 샘플 파일을 생성합니다..."
    
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
    
    log_warning "API 키를 .env 파일에 설정해주세요"
fi

# =============================================================================
# 7. Ollama 확인 (선택사항)
# =============================================================================

log_info "Ollama 서버를 확인합니다..."

if command -v ollama &> /dev/null; then
    if pgrep -f ollama > /dev/null; then
        log_success "Ollama 서버가 실행 중입니다"
        
        # 사용 가능한 모델 확인
        OLLAMA_MODELS=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
        if [ "$OLLAMA_MODELS" -gt 0 ]; then
            log_success "Ollama 모델 $OLLAMA_MODELS개 사용 가능"
        else
            log_warning "Ollama 모델이 설치되지 않았습니다"
            log_info "모델 설치 예시: ollama pull llama3.1:8b"
        fi
    else
        log_warning "Ollama 서버가 실행되지 않았습니다"
        log_info "Ollama 시작: ollama serve"
    fi
else
    log_warning "Ollama가 설치되지 않았습니다"
    log_info "Ollama 설치: https://ollama.ai"
fi

# =============================================================================
# 8. 시스템 시작
# =============================================================================

log_info "통합 RAG 시스템을 시작합니다..."

# 백그라운드 프로세스 정리 함수
cleanup() {
    log_info "시스템을 종료합니다..."
    
    # Streamlit 프로세스 종료
    pkill -f streamlit 2>/dev/null || true
    
    log_success "시스템 종료 완료"
    exit 0
}

# 시그널 핸들러 등록
trap cleanup SIGINT SIGTERM

# 로그 파일 설정
LOG_FILE="logs/unified_rag_$(date +%Y%m%d_%H%M%S).log"

log_success "시스템 구성 완료!"
echo
echo "=============================================================================="
echo -e "${GREEN}🚀 통합 RAG 시스템${NC}"
echo "=============================================================================="
echo "• 웹 인터페이스: http://localhost:8501"
echo "• Elasticsearch: http://localhost:9200"
echo "• 로그 파일: $LOG_FILE"
echo "• 종료: Ctrl+C"
echo "=============================================================================="
echo

# Streamlit 앱 실행
log_info "Streamlit 앱을 시작합니다..."

# 환경변수 설정
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Streamlit 실행 (로그와 함께)
streamlit run src/unified_rag_system.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    2>&1 | tee "$LOG_FILE"
