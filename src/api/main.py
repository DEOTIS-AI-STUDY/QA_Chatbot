#!/usr/bin/env python3
"""
FastAPI 기반 통합 RAG 시스템
- CLI와 Streamlit의 core 로직 재사용
- RESTful API 제공
- 비동기 처리 지원
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# 기존 모듈 임포트
from core.config import LLM_MODELS, HUGGINGFACE_EMBEDDINGS_AVAILABLE, UPSTAGE_AVAILABLE, OLLAMA_AVAILABLE
from core.models import ModelFactory
from core.rag import create_rag_chain
from core.chat_history import ChatHistoryManager
from utils.elasticsearch import ElasticsearchManager

# 전역 RAG 시스템 인스턴스
rag_system = None


class RAGSystemManager:
    """RAG 시스템 관리자 - FastAPI용"""
    
    def __init__(self):
        self.es_manager = None
        self.model_factory = ModelFactory()
        self.rag_chain = None
        self.embedding_model = None
        self.llm_model = None
        self.model_choice = None
        self.top_k = 5
        self.is_initialized = False
        self.initialization_time = None
        
        # 세션별 대화 기록 관리 (메모리 기반)
        self.session_managers = {}
    
    def get_chat_manager(self, session_id: str = "default") -> ChatHistoryManager:
        """세션별 대화 기록 관리자 반환"""
        if session_id not in self.session_managers:
            self.session_managers[session_id] = ChatHistoryManager(max_history=10)
        return self.session_managers[session_id]
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """의존성 확인 (비동기)"""
        issues = []
        
        if not HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            issues.append("HuggingFace 임베딩 라이브러리가 설치되지 않았습니다.")
        
        # Elasticsearch 서버 연결 확인
        try:
            es_manager = ElasticsearchManager()
            is_connected, connection_msg = es_manager.check_connection()
            if not is_connected:
                issues.append(f"Elasticsearch 서버 연결 실패: {connection_msg}")
        except Exception as e:
            issues.append(f"Elasticsearch 연결 오류: {str(e)}")
        
        # Ollama 서버 연결 확인
        try:
            from core.rag import check_ollama_connection
            ollama_connected, ollama_message = check_ollama_connection()
            if not ollama_connected:
                issues.append(f"Ollama 서버 연결 실패: {ollama_message}")
        except Exception as e:
            issues.append(f"Ollama 연결 확인 오류: {str(e)}")
        
        return {
            "status": "ok" if not issues else "error",
            "issues": issues,
            "available_models": self.model_factory.get_available_models()
        }
    
    async def initialize_rag_system(self, model_choice: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 시스템 초기화 (비동기)"""
        start_time = time.time()
        
        try:
            # 1. Elasticsearch 연결
            self.es_manager = ElasticsearchManager()
            is_connected, connection_msg = self.es_manager.check_connection()
            if not is_connected:
                raise Exception(f"Elasticsearch 연결 실패: {connection_msg}")
            
            # 2. 임베딩 모델 로드
            self.embedding_model = self.model_factory.create_embedding_model()
            if not self.embedding_model:
                raise Exception("임베딩 모델 로드 실패")
            
            # 3. LLM 모델 로드
            self.llm_model, status = self.model_factory.create_llm_model(model_choice)
            if not self.llm_model:
                raise Exception(f"LLM 모델 로드 실패: {status}")
            
            # 4. RAG 체인 생성
            self.rag_chain, success = create_rag_chain(
                embeddings=self.embedding_model,
                llm_model=self.llm_model,
                top_k=top_k
            )
            if not self.rag_chain:
                raise Exception(f"RAG 체인 생성 실패: {success}")
            
            self.model_choice = model_choice
            self.top_k = top_k
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "RAG 시스템 초기화 완료",
                "model": LLM_MODELS[model_choice]['name'],
                "top_k": top_k,
                "initialization_time": self.initialization_time
            }
            
        except Exception as e:
            self.is_initialized = False
            return {
                "status": "error",
                "message": f"RAG 시스템 초기화 실패: {str(e)}"
            }
    
    async def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """질의 처리 (비동기)"""
        if not self.is_initialized or not self.rag_chain:
            raise HTTPException(status_code=400, detail="RAG 시스템이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 대화 기록 컨텍스트 구성
            chat_manager = self.get_chat_manager(session_id)
            context_query = chat_manager.build_context_query(query)
            
            # RAG 체인 실행
            response = self.rag_chain({"query": context_query})
            
            # 응답 처리
            answer = self._extract_answer(response)
            processing_time = time.time() - start_time
            
            if answer:
                # 대화 기록에 추가
                chat_manager.add_chat(query, answer)
                
                return {
                    "status": "success",
                    "query": query,
                    "answer": answer,
                    "processing_time": processing_time,
                    "session_id": session_id,
                    "chat_history_count": chat_manager.get_history_count(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "답변을 생성할 수 없습니다.",
                    "processing_time": processing_time
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"질의 처리 오류: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _extract_answer(self, response) -> Optional[str]:
        """응답에서 답변 추출"""
        if isinstance(response, dict):
            if 'result' in response:
                return response['result']
            elif 'answer' in response:
                return response['answer']
            else:
                return str(response)
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "is_initialized": self.is_initialized,
            "model": LLM_MODELS.get(self.model_choice, {}).get('name') if self.model_choice else None,
            "model_key": self.model_choice,
            "top_k": self.top_k,
            "initialization_time": self.initialization_time,
            "active_sessions": len(self.session_managers),
            "available_models": self.model_factory.get_available_models()
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    global rag_system
    rag_system = RAGSystemManager()
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


# Pydantic 모델 정의
class QueryRequest(BaseModel):
    query: str = Field(..., description="질문 내용", min_length=1)
    session_id: str = Field(default="default", description="세션 ID")

class InitRequest(BaseModel):
    model: str = Field(..., description="사용할 LLM 모델")
    top_k: int = Field(default=5, description="검색 결과 상위 K개", ge=1, le=20)

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    count: int
    session_id: str


# API 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "통합 RAG 시스템 API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/dependencies")
async def check_dependencies():
    """의존성 확인"""
    result = await rag_system.check_dependencies()
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result)
    return result

@app.get("/models")
async def get_available_models():
    """사용 가능한 모델 목록"""
    return {
        "models": rag_system.model_factory.get_available_models(),
        "status": "success"
    }

@app.post("/initialize")
async def initialize_system(request: InitRequest):
    """RAG 시스템 초기화"""
    available_models = rag_system.model_factory.get_available_models()
    
    if request.model not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"모델 '{request.model}'를 찾을 수 없습니다. 사용 가능한 모델: {list(available_models.keys())}"
        )
    
    result = await rag_system.initialize_rag_system(request.model, request.top_k)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@app.get("/status")
async def get_system_status():
    """시스템 상태 확인"""
    return rag_system.get_system_info()

@app.post("/query")
async def process_query(request: QueryRequest):
    """질의 처리"""
    result = await rag_system.process_query(request.query, request.session_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str = "default"):
    """대화 기록 조회"""
    chat_manager = rag_system.get_chat_manager(session_id)
    
    return ChatHistoryResponse(
        history=chat_manager.get_history(),
        count=chat_manager.get_history_count(),
        session_id=session_id
    )

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str = "default"):
    """대화 기록 삭제"""
    chat_manager = rag_system.get_chat_manager(session_id)
    chat_manager.clear_history()
    
    return {
        "status": "success",
        "message": f"세션 '{session_id}'의 대화 기록이 삭제되었습니다.",
        "session_id": session_id
    }

@app.get("/sessions")
async def get_active_sessions():
    """활성 세션 목록"""
    return {
        "sessions": list(rag_system.session_managers.keys()),
        "count": len(rag_system.session_managers)
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id in rag_system.session_managers:
        del rag_system.session_managers[session_id]
        return {
            "status": "success",
            "message": f"세션 '{session_id}'가 삭제되었습니다."
        }
    else:
        raise HTTPException(status_code=404, detail=f"세션 '{session_id}'를 찾을 수 없습니다.")


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
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 활성화")
    
    args = parser.parse_args()
    
    print(f"🚀 FastAPI RAG 서버 시작: http://{args.host}:{args.port}")
    print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
