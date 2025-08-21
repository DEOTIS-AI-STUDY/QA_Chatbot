#!/usr/bin/env python3
"""
FastAPI 기반 통합 RAG 시스템
- unified_rag_cli.py와 동일한 core 모듈 import 방식 사용
- RESTful API 제공
- 비동기 처리 지원
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    # 프로젝트 루트에서 .env.prod 파일 로드
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env.prod')
    load_dotenv(env_path)
    print(f"🔧 환경 변수 로드: {env_path}")
    print(f"🔗 OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'Not set')}")
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않았습니다. 환경 변수를 수동으로 설정해주세요.")

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# unified_rag_cli.py와 동일한 모듈 import 방식
from core.config import LLM_MODELS, HUGGINGFACE_EMBEDDINGS_AVAILABLE, OLLAMA_AVAILABLE
from core.models import ModelFactory
from core.rag import create_llm_chain, create_rag_chain, create_retriever, prompt_for_refined_query, prompt_for_query, prompt_for_context_summary
from core.chat_history import ChatHistoryManager
from utils.elasticsearch import ElasticsearchManager

# langfuse_config import - 로컬 실행과 Docker 모듈 실행 모두 지원
try:
    # Docker에서 모듈로 실행할 때
    from .langfuse_config import get_langfuse_manager, get_langfuse_callback
except ImportError:
    # 로컬에서 직접 실행할 때
    from langfuse_config import get_langfuse_manager, get_langfuse_callback

# Elasticsearch 가용성 확인
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠️ Elasticsearch가 설치되지 않았습니다. pip install elasticsearch")

# 전역 RAG 시스템 인스턴스
rag_system = None


class FastAPIRAGSystem:
    """FastAPI용 RAG 시스템 - unified_rag_cli.py와 동일한 로직"""
    
    def __init__(self):
        self.es_manager = None
        self.model_factory = ModelFactory()
        self.rag_chain = None
        self.embedding_model = None
        self.llm_model = None
        self.model_choice = None
        self.top_k = 5
        self.retriever = None
        self.llm_chain = None
        
        # Langfuse 매니저 초기화
        self.langfuse_manager = get_langfuse_manager()
        
        # 세션별 대화 기록 관리 (메모리 기반)
        self.session_managers = {}
        self.is_initialized = False
        self.initialization_time = None
    
    def get_chat_manager(self, session_id: str = "default") -> ChatHistoryManager:
        """세션별 대화 기록 관리자 반환"""
        if session_id not in self.session_managers:
            self.session_managers[session_id] = ChatHistoryManager(max_history=10)
        return self.session_managers[session_id]
    
    async def check_dependencies_async(self) -> Dict[str, Any]:
        """의존성 확인 (비동기 버전)"""
        def _check_dependencies():
            issues = []
            
            if not ELASTICSEARCH_AVAILABLE:
                issues.append("Elasticsearch 라이브러리가 설치되지 않았습니다.")
            
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
                "issue_count": len(issues)
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, _check_dependencies)
    
    async def initialize_rag_system_async(self, model_choice: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 시스템 초기화 (비동기 버전)"""
        def _initialize_rag_system():
            try:
                start_time = time.time()
                
                self.model_choice = model_choice
                self.top_k = top_k
                
                # Elasticsearch 관리자 초기화
                self.es_manager = ElasticsearchManager()
                
                # 임베딩 모델 로드
                self.embedding_model = self.model_factory.create_embedding_model()
                if not self.embedding_model:
                    return {
                        "status": "error",
                        "message": "임베딩 모델 로드 실패"
                    }
                
                # LLM 모델 로드
                self.llm_model, status = self.model_factory.create_llm_model(model_choice)
                if not self.llm_model:
                    return {
                        "status": "error",
                        "message": f"LLM 모델 로드 실패: {status}"
                    }
                
                # RAG 체인 생성 (Langfuse 콜백 포함)
                langfuse_callback = get_langfuse_callback()
                callbacks = [langfuse_callback] if langfuse_callback else None
                
                self.rag_chain, success_or_error = create_rag_chain(
                    embeddings=self.embedding_model,
                    llm_model=self.llm_model,
                    top_k=top_k,
                    callbacks=callbacks
                )

                # Retriever 생성
                self.retriever = create_retriever(
                    embedding_model=self.embedding_model,
                    top_k=top_k
                )


                # LLM 체인 생성
                try:
                    self.llm_chain = create_llm_chain(
                        llm_model=self.llm_model,
                        prompt_template="""{context}, {question}"""
                    )
                except Exception as e:
                    print(f"❌ LLM 체인 생성 오류: {str(e)}")
                    self.llm_chain = None

                if self.rag_chain:
                    self.is_initialized = True
                    self.initialization_time = time.time() - start_time
                    return {
                        "status": "success",
                        "message": "RAG 시스템 초기화 완료",
                        "model": LLM_MODELS[model_choice]['name'],
                        "top_k": top_k,
                        "initialization_time": self.initialization_time
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"RAG 체인 생성 실패: {success_or_error}"
                    }
                    
                    
            except Exception as e:
                self.is_initialized = False
                return {
                    "status": "error",
                    "message": f"RAG 시스템 초기화 실패: {str(e)}"
                }
        
        return await asyncio.get_event_loop().run_in_executor(None, _initialize_rag_system)
    
    async def process_query_async(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """질의 처리 (비동기 버전)"""
        if not self.is_initialized or not self.rag_chain:
            return {
                "status": "error",
                "message": "RAG 시스템이 초기화되지 않았습니다."
            }

        def _process_query():
            try:
                start_time = time.time()

                # Langfuse 트레이스 생성
                trace = self.langfuse_manager.create_trace(
                    name="rag_query",
                    metadata={
                        "session_id": session_id,
                        "model": self.model_choice,
                        "top_k": self.top_k
                    }
                )

                # Langfuse 콜백 준비
                langfuse_callback = None
                try:
                    from core.rag import get_langfuse_callback
                    langfuse_callback = get_langfuse_callback()
                except Exception:
                    pass
                callbacks = [langfuse_callback] if langfuse_callback else None

                # 대화 기록 관리자 가져오기
                chat_manager = self.get_chat_manager(session_id)

                # 대화 기록으로 질문 재정의.
                history = chat_manager.build_history()
                print(f"🔍 대화 기록: {history}")
                print(f"🔍 질의: {query}")
                llm_chain_refine = create_llm_chain(self.llm_model, prompt_for_refined_query, callbacks=callbacks)
                refined_query = llm_chain_refine.run({"question": query, "context": history})
                print(f"🔍 정제된 질의: {refined_query}")

                # 재정의된 질문으로 DB 검색
                docs = self.retriever.get_relevant_documents(refined_query)
                docs_text = "\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])
                print(f"🔍 검색된 문서 내용: {docs_text}")

                # 검색된 자료와 재정의 질문을 LLM에 넘겨서 답변 생성
                llm_chain_query = create_llm_chain(self.llm_model, prompt_for_query, callbacks=callbacks)
                result = llm_chain_query.invoke({"question": refined_query, "context": docs_text})

                # RAG 체인을 통한 답변 생성
                #result = self.rag_chain.invoke({"query": query})

                # 디버깅: 실제 응답 구조 출력
                print(f"🔍 RAG 체인 응답 구조: {result}")
                print(f"🔍 응답 키들: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

                processing_time = time.time() - start_time

                # RetrievalQA는 'result' 키를 사용함
                if result and ('answer' in result or 'result' in result or 'text' in result):
                    answer = result.get('answer') or result.get('result') or result.get('text')
                    print(f"🔍 최종 답변: {answer}")
                    # 답변 요약
                    llm_chain_summary = create_llm_chain(self.llm_model, prompt_for_context_summary, callbacks=callbacks)
                    answer_summary = llm_chain_summary.run({"context": answer})
                    print(f"🔍 답변 요약: {answer_summary}")
                    # 대화 기록에 질문과 답변 추가
                    chat_manager.add_chat(refined_query, answer_summary)

                    # Langfuse에 결과 로그
                    if trace:
                        self.langfuse_manager.log_generation(
                            trace_context=trace.get('trace_context'),
                            name="rag_generation",
                            input=query,
                            output=answer,
                            metadata={
                                "processing_time": processing_time,
                                "retrieved_docs_count": len(result.get('source_documents', [])),
                                "model": self.model_choice
                            }
                        )

                    return {
                        "status": "success",
                        "answer": answer,
                        "query": query,
                        "session_id": session_id,
                        "processing_time": processing_time,
                        "retrieved_docs": result.get('source_documents', [])
                    }
                else:
                    # Langfuse에 에러 로그
                    if trace:
                        self.langfuse_manager.log_event(
                            trace_context=trace.get('trace_context'),
                            name="rag_error",
                            metadata={
                                "error": f"답변 생성 실패. 응답 구조: {result}",
                                "processing_time": processing_time
                            }
                        )

                    return {
                        "status": "error",
                        "message": f"답변을 생성할 수 없습니다. 응답 구조: {result}",
                        "processing_time": processing_time
                    }

            except Exception as e:
                # Langfuse에 예외 로그
                if 'trace' in locals() and trace:
                    self.langfuse_manager.log_event(
                        trace_context=trace.get('trace_context'),
                        name="rag_exception",
                        metadata={
                            "error": str(e),
                            "processing_time": time.time() - start_time
                        }
                    )

                return {
                    "status": "error",
                    "message": f"질의 처리 오류: {str(e)}",
                    "processing_time": time.time() - start_time
                }

        return await asyncio.get_event_loop().run_in_executor(None, _process_query)
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "is_initialized": self.is_initialized,
            "model": LLM_MODELS.get(self.model_choice, {}).get('name') if self.model_choice else None,
            "model_key": self.model_choice,
            "top_k": self.top_k,
            "initialization_time": self.initialization_time,
            "active_sessions": len(self.session_managers),
            "available_models": self.model_factory.get_available_models(),
            "langfuse_status": self.langfuse_manager.get_status()
        }


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

class LangfuseTraceResponse(BaseModel):
    data: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]] = None

class LangfuseUsageResponse(BaseModel):
    total_tokens: int
    total_cost: float
    model_usage: Dict[str, Dict[str, Any]]
    observation_count: int


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
    result = await rag_system.check_dependencies_async()
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
    
    result = await rag_system.initialize_rag_system_async(request.model, request.top_k)
    
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
    result = await rag_system.process_query_async(request.query, request.session_id)
    
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


@app.get("/langfuse/status", tags=["Langfuse"])
async def get_langfuse_status():
    """Langfuse 상태 확인"""
    return rag_system.langfuse_manager.get_status()


@app.post("/langfuse/flush", tags=["Langfuse"])
async def flush_langfuse():
    """Langfuse 로그 강제 전송"""
    try:
        rag_system.langfuse_manager.flush()
        return {
            "status": "success",
            "message": "Langfuse 로그가 전송되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Langfuse flush 실패: {str(e)}")


@app.get("/langfuse/traces", tags=["Langfuse"], response_model=Dict[str, Any])
async def get_langfuse_traces(
    limit: int = 50,
    page: int = 1,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Langfuse 트레이스 목록 조회
    
    - **limit**: 페이지당 결과 수 (기본값: 50)
    - **page**: 페이지 번호 (기본값: 1)
    - **session_id**: 세션 ID 필터
    - **user_id**: 사용자 ID 필터
    """
    try:
        filters = {}
        if session_id:
            filters['session_id'] = session_id
        if user_id:
            filters['user_id'] = user_id
            
        result = rag_system.langfuse_manager.get_traces(
            limit=limit,
            page=page,
            **filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"트레이스 조회 실패: {str(e)}")


@app.get("/langfuse/traces/{trace_id}", tags=["Langfuse"])
async def get_langfuse_trace(trace_id: str):
    """특정 트레이스 상세 조회
    
    - **trace_id**: 조회할 트레이스 ID
    """
    try:
        result = rag_system.langfuse_manager.get_trace_by_id(trace_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"트레이스 조회 실패: {str(e)}")


@app.get("/langfuse/observations", tags=["Langfuse"])
async def get_langfuse_observations(
    limit: int = 50,
    page: int = 1,
    trace_id: Optional[str] = None,
    type: Optional[str] = None
):
    """Langfuse 관찰 데이터 조회 (LLM 호출, 스팬 등)
    
    - **limit**: 페이지당 결과 수 (기본값: 50)
    - **page**: 페이지 번호 (기본값: 1)
    - **trace_id**: 트레이스 ID 필터
    - **type**: 관찰 타입 필터 (GENERATION, SPAN, EVENT 등)
    """
    try:
        filters = {}
        if trace_id:
            filters['trace_id'] = trace_id
        if type:
            filters['type'] = type
            
        result = rag_system.langfuse_manager.get_observations(
            limit=limit,
            page=page,
            **filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"관찰 데이터 조회 실패: {str(e)}")


@app.get("/langfuse/sessions", tags=["Langfuse"])
async def get_langfuse_sessions(
    limit: int = 50,
    page: int = 1
):
    """Langfuse 세션 목록 조회
    
    - **limit**: 페이지당 결과 수 (기본값: 50)
    - **page**: 페이지 번호 (기본값: 1)
    """
    try:
        result = rag_system.langfuse_manager.get_sessions(
            limit=limit,
            page=page
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 조회 실패: {str(e)}")


@app.get("/langfuse/usage", tags=["Langfuse"], response_model=LangfuseUsageResponse)
async def get_langfuse_usage(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """LLM 사용 통계 조회
    
    - **from_date**: 시작 날짜 (ISO 형식, 예: 2023-01-01T00:00:00Z)
    - **to_date**: 종료 날짜 (ISO 형식, 예: 2023-12-31T23:59:59Z)
    
    반환 데이터:
    - total_tokens: 총 토큰 사용량
    - total_cost: 총 비용
    - model_usage: 모델별 사용 통계
    - observation_count: 총 관찰 데이터 수
    """
    try:
        result = rag_system.langfuse_manager.get_usage_statistics(
            from_date=from_date,
            to_date=to_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용 통계 조회 실패: {str(e)}")


@app.get("/langfuse/metrics", tags=["Langfuse"])
async def get_langfuse_metrics():
    """Langfuse 메트릭 조회
    
    시스템 전체 메트릭 및 성능 지표를 조회합니다.
    """
    try:
        result = rag_system.langfuse_manager.get_metrics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메트릭 조회 실패: {str(e)}")


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
    from core.config import PDF_DIR, INDEX_NAME
    from elasticsearch import Elasticsearch
    
    parser = argparse.ArgumentParser(description="FastAPI RAG 시스템 서버")
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8110, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 활성화")
    parser.add_argument("--init-index", action="store_true", help="PDF 자동 인덱싱만 수행하고 종료")
    args = parser.parse_args()

    def auto_index_pdfs():
        from core.models import ModelFactory
        from utils.elasticsearch import ElasticsearchManager
        import glob
        import time
        print(f"\n📄 PDF 자동 인덱싱 시작 (INDEX_NAME: {INDEX_NAME})")
        if not os.path.exists(PDF_DIR):
            print(f"📁 PDF 디렉토리({PDF_DIR})가 없습니다. 생성합니다...")
            os.makedirs(PDF_DIR, exist_ok=True)
            print("📄 PDF 파일이 없어서 인덱싱을 건너뜁니다.")
            return
        pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
        if not pdf_files:
            print("📄 PDF 파일이 없어서 인덱싱을 건너뜁니다.")
            return
        try:
            es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"), timeout=10)
            if es.indices.exists(index=INDEX_NAME):
                doc_count = es.count(index=INDEX_NAME).get("count", 0)
                if doc_count > 0:
                    print(f"📚 기존 인덱스에 {doc_count}개 문서가 있습니다. 인덱싱을 건너뜁니다.")
                    return
        except Exception as check_error:
            print(f"⚠️ 인덱스 확인 중 오류: {str(check_error)}, 인덱싱을 계속 진행합니다...")
        print(f"📄 {len(pdf_files)}개 PDF 파일을 자동 인덱싱합니다...")
        for pdf_file in pdf_files:
            print(f"  - {os.path.basename(pdf_file)}")
        try:
            class SimpleTracker:
                def track_preprocessing_stage(self, stage):
                    print(f"🔄 {stage}")
                def end_preprocessing_stage(self, stage):
                    print(f"✅ {stage} 완료")
            tracker = SimpleTracker()
            es_manager = ElasticsearchManager()
            embedding_model = ModelFactory().create_embedding_model()
            indexing_start = time.time()
            success, message = es_manager.index_pdfs(pdf_files, embedding_model, tracker)
            indexing_time = time.time() - indexing_start
            if success:
                print(f"✅ PDF 자동 인덱싱 완료: {message} ({indexing_time:.2f}초)")
            else:
                print(f"❌ PDF 자동 인덱싱 실패: {message}")
        except Exception as e:
            print(f"❌ PDF 자동 인덱싱 오류: {str(e)}")

    if args.init_index:
        auto_index_pdfs()
        print("\n✅ --init-index 작업이 완료되었습니다. 서버는 실행하지 않습니다.")
        sys.exit(0)

    print(f"🚀 FastAPI RAG 서버 시작: http://{args.host}:{args.port}")
    print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
