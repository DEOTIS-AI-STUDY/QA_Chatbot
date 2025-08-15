#!/bin/bash

# =============================================================================
# 간편 실행 스크립트 - Docker와 Elasticsearch 자동 시작
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
echo -e "${GREEN}🚀 통합 RAG 시스템 간편 실행${NC}"
echo "=============================================================================="

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Docker 상태 확인 및 자동 시작
log_info "Docker 상태를 확인합니다..."

if ! docker info &> /dev/null; then
    log_warning "Docker 데몬이 실행되지 않고 있습니다."
    
    # macOS에서 Docker Desktop 자동 시작
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/Applications/Docker.app" ]; then
            log_info "Docker Desktop을 시작합니다..."
            open -a Docker
            
            log_info "Docker Desktop 시작 대기 중... (최대 60초)"
            for i in {1..12}; do
                if docker info &> /dev/null; then
                    log_success "Docker 데몬이 시작되었습니다."
                    break
                fi
                printf "\r${BLUE}[INFO]${NC} Docker 데몬 시작 대기 중... (%d/12)" $i
                sleep 5
            done
            echo # 줄바꿈
            
            if ! docker info &> /dev/null; then
                log_error "Docker 데몬 시작에 실패했습니다."
                echo "다음 중 하나를 시도해보세요:"
                echo "1. Docker Desktop을 수동으로 시작"
                echo "2. 시스템 재시작 후 다시 시도"
                echo "3. Docker Desktop 재설치"
                exit 1
            fi
        else
            log_error "Docker Desktop이 설치되지 않았습니다."
            echo "Docker Desktop을 설치해주세요: https://docs.docker.com/desktop/mac/install/"
            exit 1
        fi
    else
        log_error "이 스크립트는 현재 macOS만 지원합니다."
        echo "다른 운영체제에서는 Docker를 수동으로 시작한 후:"
        echo "./run_unified_rag_refactored.sh --no-check"
        exit 1
    fi
else
    log_success "Docker 데몬이 이미 실행 중입니다."
fi

# 2. Elasticsearch 자동 시작
log_info "Elasticsearch를 확인합니다..."

ES_URL="http://localhost:9200"

if curl -s "$ES_URL" > /dev/null 2>&1; then
    log_success "Elasticsearch가 이미 실행 중입니다."
else
    log_info "Elasticsearch를 시작합니다..."
    
    # Docker Compose로 Elasticsearch 시작
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d elasticsearch
        
        log_info "Elasticsearch 시작 대기 중... (최대 60초)"
        for i in {1..30}; do
            if curl -s "$ES_URL" > /dev/null 2>&1; then
                log_success "Elasticsearch가 시작되었습니다."
                break
            fi
            printf "\r${BLUE}[INFO]${NC} Elasticsearch 시작 대기 중... (%d/30)" $i
            sleep 2
        done
        echo # 줄바꿈
        
        if ! curl -s "$ES_URL" > /dev/null 2>&1; then
            log_error "Elasticsearch 시작에 실패했습니다."
            echo "수동으로 확인해보세요:"
            echo "  docker-compose logs elasticsearch"
            echo "  docker-compose restart elasticsearch"
            exit 1
        fi
    else
        log_error "docker-compose.yml 파일을 찾을 수 없습니다."
        exit 1
    fi
fi

# 3. 포트 충돌 사전 확인
log_info "포트 8501 사용 상태를 확인합니다..."

if lsof -ti :8501 >/dev/null 2>&1; then
    log_warning "포트 8501이 이미 사용 중입니다."
    
    # Docker 컨테이너가 포트를 사용 중인지 확인
    docker_container=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep ":8501->" | awk '{print $1}')
    
    if [ -n "$docker_container" ]; then
        log_info "Docker 컨테이너 '$docker_container'가 포트 8501을 사용 중입니다."
        log_info "기존 컨테이너를 중지하고 새로운 인스턴스를 시작합니다..."
        
        docker stop "$docker_container" > /dev/null 2>&1
        docker rm "$docker_container" > /dev/null 2>&1
        log_success "기존 Docker 컨테이너가 중지되었습니다."
    else
        # 일반 프로세스가 포트 사용 중
        process_info=$(lsof -ti :8501)
        log_warning "PID $process_info가 포트 8501을 사용 중입니다."
        log_info "해당 프로세스를 종료합니다..."
        
        kill -9 $process_info 2>/dev/null || true
        sleep 2
        log_success "프로세스가 종료되었습니다."
    fi
else
    log_success "포트 8501은 사용 가능합니다."
fi

# 4. RAG 시스템 실행
log_success "모든 준비가 완료되었습니다!"
echo
log_info "리팩토링된 통합 RAG 시스템을 시작합니다..."

# 리팩토링된 스크립트 실행 (체크 없이)
if [ -f "run_unified_rag_refactored.sh" ]; then
    ./run_unified_rag_refactored.sh --no-check
else
    log_error "run_unified_rag_refactored.sh 파일을 찾을 수 없습니다."
    echo "다음을 확인해주세요:"
    echo "1. 올바른 디렉토리에서 실행 중인지 확인"
    echo "2. run_unified_rag_refactored.sh 파일이 존재하는지 확인"
    echo "3. 실행 권한이 있는지 확인: chmod +x run_unified_rag_refactored.sh"
    exit 1
fi
