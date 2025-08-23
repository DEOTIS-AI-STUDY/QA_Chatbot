#!/usr/bin/env python3
"""
FastAPI 기반 통합 RAG 시스템
- 모듈화된 구조 사용
- RESTful API 제공
- 비동기 처리 지원
"""

import os
import sys
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI 관련 import
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 현재 디렉토리를 모듈 검색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 핵심 기능 모듈들
from config.app_config import (
    load_environment, 
    FastAPIRAGSystem,
    get_langfuse_manager,
    get_langfuse_callback,
    LLM_MODELS,
    HUGGINGFACE_EMBEDDINGS_AVAILABLE
)
from core.config import INDEX_NAME
from utils.file_utils import auto_index_files

# 라우트 모듈 import
from routes import create_endpoints
from routes.endpoint import get_rag_system

# 환경 설정 로드
load_environment()

# 전역 RAG 시스템 인스턴스
rag_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    global rag_system
    rag_system = FastAPIRAGSystem()
    print("🚀 FastAPI RAG 시스템 시작")
    
    yield
    
    # 종료 시
    print("👋 FastAPI RAG 시스템 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="통합 RAG 시스템 API",
    description="BGE-M3 임베딩 + Elasticsearch + 멀티 LLM을 사용한 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
current_api_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=current_api_dir), name="static")

# RAG 시스템 의존성 함수 오버라이드
def override_get_rag_system():
    """RAG 시스템 의존성 주입 함수 오버라이드"""
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")
    return rag_system

# 의존성 함수 오버라이드
import routes.endpoint
import routes.endpoint_langfuse
routes.endpoint.get_rag_system = override_get_rag_system
routes.endpoint_langfuse.get_rag_system = override_get_rag_system

# 엔드포인트 생성
create_endpoints(app, current_api_dir)

# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"서버 내부 오류: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI RAG 시스템 서버")
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8110, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 활성화")
    parser.add_argument("--init-index", action="store_true", help="파일 자동 인덱싱만 수행하고 종료")
    parser.add_argument("--file-types", nargs='+', default=['pdf'], 
                       choices=['pdf', 'txt', 'json', 'docx', 'all'], 
                       help="인덱싱할 파일 타입 선택 (기본값: pdf)")
    args = parser.parse_args()

    # 파일 자동 인덱싱 실행 (import된 함수 사용)
    if args.init_index:
        print("🚀 파일 자동 인덱싱을 실행합니다...")
        auto_index_files(args.file_types, INDEX_NAME)
        print("✅ 파일 자동 인덱싱이 완료되었습니다.")
        exit(0)
    
    # FastAPI 서버 실행
    print(f"🚀 FastAPI RAG 서버 시작: http://{args.host}:{args.port}")
    print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
