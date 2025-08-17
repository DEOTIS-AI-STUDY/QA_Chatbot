#!/bin/bash

# =============================================================================
# CLI RAG 시스템 간편 실행 스크립트
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
echo -e "${GREEN}🤖 CLI 기반 통합 RAG 시스템${NC}"
echo "=============================================================================="

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 가상환경 활성화
if [ -f ".venv/bin/activate" ]; then
    log_info "Python 가상환경을 활성화합니다..."
    source .venv/bin/activate
else
    log_warning "가상환경을 찾을 수 없습니다. 시스템 Python을 사용합니다."
fi

# 기본 설정
DEFAULT_MODEL="solar_10_7b"
DEFAULT_TOP_K=5

# 명령행 인자 처리
MODEL=""
TOP_K=""
SHOW_HELP=false

# 인자 파싱
for arg in "$@"; do
    case $arg in
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --top-k=*)
            TOP_K="${arg#*=}"
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
        *)
            log_error "알 수 없는 인자: $arg"
            echo "사용법: $0 [--model=모델명] [--top-k=값] [--help]"
            exit 1
            ;;
    esac
done

# 도움말 표시
if [ "$SHOW_HELP" = true ]; then
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  --model=MODEL     사용할 모델 (solar_10_7b, qwen2, llama3)"
    echo "  --top-k=K         검색 결과 상위 K개 (기본값: 5)"
    echo "  --help, -h        이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0                          # 인터랙티브 모드로 설정 선택"
    echo "  $0 --model=solar_10_7b      # 특정 모델 사용"
    echo "  $0 --top-k=10               # Top-K 값 설정"
    echo "  $0 --model=qwen2 --top-k=7  # 모델과 Top-K 모두 설정"
    exit 0
fi

PYTHON_CMD="python src/unified_rag_cli.py"

# 모델 선택 (명령행 인자가 없을 경우에만)
if [ -z "$MODEL" ]; then
    echo
    echo "사용할 모델을 선택하세요:"
    echo "1. SOLAR-10.7B"
    echo "2. Qwen2"
    echo "3. Llama3"
    echo
    
    read -p "모델 번호 입력 (1-3): " model_choice
    
    case $model_choice in
        1) MODEL="solar_10_7b" ;;
        2) MODEL="qwen2" ;;
        3) MODEL="llama3" ;;
        *) MODEL="$DEFAULT_MODEL" ;;
    esac
fi

# Top-K 값 선택 (명령행 인자가 없을 경우에만)
if [ -z "$TOP_K" ]; then
    echo
    read -p "Top-K 값을 입력하세요 (기본값: 5): " top_k_input
    TOP_K="${top_k_input:-$DEFAULT_TOP_K}"
fi

log_success "선택된 모델: $MODEL"
log_success "Top-K 값: $TOP_K"

log_success "선택된 모델: $MODEL"
log_success "Top-K 값: $TOP_K"

# 대화형 모드로 실행
log_info "대화형 모드로 시작합니다..."
$PYTHON_CMD --model "$MODEL" --top-k "$TOP_K"

log_success "CLI RAG 시스템 실행이 완료되었습니다."
