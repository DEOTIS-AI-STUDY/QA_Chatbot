#!/bin/bash

# =============================================================================
# 리팩토링된 통합 RAG 시스템 실행 스크립트
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

# 프로젝트 루트 디렉토리 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

log_info "프로젝트 루트: $PROJECT_ROOT"

# Python path 설정
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 포트 충돌 해결 함수
resolve_port_conflict() {
    local port=${1:-8501}
    log_info "포트 $port 사용 상태를 확인합니다..."
    
    # 포트 사용 중인 프로세스 확인
    local process_info=$(lsof -ti :$port 2>/dev/null)
    
    if [ -n "$process_info" ]; then
        log_warning "포트 $port가 이미 사용 중입니다."
        
        # Docker 컨테이너가 포트를 사용 중인지 확인
        local docker_container=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep ":$port->" | awk '{print $1}')
        
        if [ -n "$docker_container" ]; then
            log_info "Docker 컨테이너 '$docker_container'가 포트 $port를 사용 중입니다."
            
            read -p "Docker 컨테이너를 중지하고 계속하시겠습니까? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Docker 컨테이너 '$docker_container'를 중지합니다..."
                docker stop "$docker_container" > /dev/null 2>&1
                docker rm "$docker_container" > /dev/null 2>&1
                log_success "Docker 컨테이너가 중지되었습니다."
            else
                log_error "포트 충돌로 인해 실행을 중단합니다."
                echo "다음 중 하나를 선택하세요:"
                echo "1. Docker 컨테이너 수동 중지: docker stop $docker_container"
                echo "2. 다른 포트 사용: --server.port 8502"
                echo "3. Docker 컨테이너로 접속: http://localhost:$port"
                exit 1
            fi
        else
            # 일반 프로세스가 포트 사용 중
            log_info "PID $process_info가 포트 $port를 사용 중입니다."
            
            read -p "해당 프로세스를 종료하고 계속하시겠습니까? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "프로세스 $process_info를 종료합니다..."
                kill -9 $process_info 2>/dev/null || true
                sleep 2
                log_success "프로세스가 종료되었습니다."
            else
                log_error "포트 충돌로 인해 실행을 중단합니다."
                echo "수동으로 프로세스를 종료하거나 다른 포트를 사용하세요."
                exit 1
            fi
        fi
        
        # 포트가 해제되었는지 재확인
        sleep 2
        if lsof -ti :$port >/dev/null 2>&1; then
            log_error "포트 $port가 여전히 사용 중입니다."
            exit 1
        else
            log_success "포트 $port가 해제되었습니다."
        fi
    else
        log_success "포트 $port는 사용 가능합니다."
    fi
}

# 사전 체크 함수
check_requirements() {
    log_info "필수 요구사항 확인 중..."
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되지 않았습니다."
        exit 1
    fi
    
    # 환경변수 파일 확인
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        log_warning ".env 파일이 없습니다. 기본 설정을 사용합니다."
    fi
    
    # src 디렉토리 확인
    if [[ ! -d "$PROJECT_ROOT/src" ]]; then
        log_error "src 디렉토리가 없습니다."
        exit 1
    fi
    
    log_success "요구사항 확인 완료"
}

# 종속성 설치 함수
install_dependencies() {
    log_info "Python 종속성 설치 중..."
    
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        python3 -m pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
        log_success "requirements.txt 종속성 설치 완료"
    fi
    
    if [[ -f "$PROJECT_ROOT/requirements_unified.txt" ]]; then
        python3 -m pip install -r "$PROJECT_ROOT/requirements_unified.txt" --quiet
        log_success "requirements_unified.txt 종속성 설치 완료"
    fi
}

# Docker 및 Elasticsearch 상태 확인 및 자동 시작
check_elasticsearch() {
    log_info "Docker 및 Elasticsearch 상태 확인 중..."
    
    # Docker 데몬 상태 확인
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
    else
        log_success "Docker 데몬이 실행 중입니다."
    fi
    
    # Elasticsearch 상태 확인 및 자동 시작
    ES_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"
    
    if curl -s "$ES_URL" > /dev/null 2>&1; then
        log_success "Elasticsearch 연결 확인됨: $ES_URL"
    else
        log_warning "Elasticsearch에 연결할 수 없습니다: $ES_URL"
        log_info "Elasticsearch를 자동으로 시작합니다..."
        
        # 기존 Elasticsearch 컨테이너 확인
        if docker ps --format "table {{.Names}}" | grep -q elasticsearch; then
            log_info "Elasticsearch 컨테이너는 실행 중이지만 응답하지 않습니다. 재시작합니다..."
            docker-compose restart elasticsearch
        else
            log_info "Elasticsearch 컨테이너를 시작합니다..."
            
            # Docker Compose로 Elasticsearch 시작
            if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
                cd "$PROJECT_ROOT"
                docker-compose up -d elasticsearch
            else
                log_error "docker-compose.yml 파일을 찾을 수 없습니다."
                exit 1
            fi
        fi
        
        # Elasticsearch가 준비될 때까지 대기
        log_info "Elasticsearch가 준비될 때까지 대기 중..."
        for i in {1..30}; do
            if curl -s "$ES_URL" > /dev/null 2>&1; then
                log_success "Elasticsearch가 시작되었습니다: $ES_URL"
                break
            fi
            log_info "Elasticsearch 시작 대기 중... ($i/30)"
            sleep 2
        done
        
        # 최종 확인
        if ! curl -s "$ES_URL" > /dev/null 2>&1; then
            log_error "Elasticsearch 시작에 실패했습니다."
            log_info "수동으로 확인해주세요: docker-compose up -d elasticsearch"
            exit 1
        fi
        
        log_success "Elasticsearch가 정상적으로 시작되었습니다."
    fi
}

# 메인 실행 함수
run_app() {
    log_info "리팩토링된 통합 RAG 시스템 시작 중..."
    
    cd "$PROJECT_ROOT"
    
    # 포트 충돌 해결
    resolve_port_conflict 8501
    
    # Streamlit 실행
    python3 -m streamlit run src/unified_rag_refactored.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false
}

# 도움말 함수
show_help() {
    echo "사용법: $0 [OPTIONS]"
    echo ""
    echo "옵션:"
    echo "  --no-deps    종속성 설치 건너뛰기"
    echo "  --no-check   사전 체크 건너뛰기 (Docker/Elasticsearch 자동 시작 포함)"
    echo "  --help       이 도움말 표시"
    echo ""
    echo "기능:"
    echo "  - Docker 데몬 자동 시작 (macOS Docker Desktop)"
    echo "  - Elasticsearch 자동 시작 및 상태 확인"
    echo "  - Python 가상환경 자동 활성화"
    echo "  - 종속성 자동 설치"
    echo ""
    echo "예시:"
    echo "  $0                # 전체 체크 및 실행 (권장)"
    echo "  $0 --no-deps     # 종속성 설치 없이 실행"
    echo "  $0 --no-check    # 사전 체크 없이 실행"
}

# 메인 로직
main() {
    local skip_deps=false
    local skip_check=false
    
    # 인수 파싱
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-deps)
                skip_deps=true
                shift
                ;;
            --no-check)
                skip_check=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "=== 리팩토링된 통합 RAG 시스템 시작 ==="
    
    # 사전 체크 실행
    if [[ "$skip_check" != true ]]; then
        check_requirements
        check_elasticsearch
    fi
    
    # 종속성 설치
    if [[ "$skip_deps" != true ]]; then
        install_dependencies
    fi
    
    # 앱 실행
    run_app
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
