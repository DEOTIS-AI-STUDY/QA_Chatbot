#!/usr/bin/env python3
"""
Langfuse 관련 FastAPI 엔드포인트 정의
- Langfuse 모니터링 및 분석 엔드포인트들
- 트레이스, 관찰, 세션, 사용량 통계 등 제공
"""

from typing import Dict, Any, List, Optional
from fastapi import HTTPException, Depends

# Pydantic 모델들
from models.api_models import LangfuseUsageResponse

# 핵심 기능 모듈들
from config.app_config import FastAPIRAGSystem


def get_rag_system() -> FastAPIRAGSystem:
    """RAG 시스템 의존성 주입 함수"""
    # 이 함수는 main.py에서 덮어쓰여집니다.
    raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")


def create_langfuse_endpoints(app):
    """Langfuse 관련 엔드포인트 생성 함수"""
    
    @app.get("/langfuse/status", tags=["Langfuse"])
    async def get_langfuse_status(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """Langfuse 상태 확인"""
        return rag_system.langfuse_manager.get_status()

    @app.post("/langfuse/flush", tags=["Langfuse"])
    async def flush_langfuse(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """Langfuse 데이터 강제 전송"""
        try:
            rag_system.langfuse_manager.flush()
            return {"status": "success", "message": "Langfuse 데이터가 플러시되었습니다."}
        except Exception as e:
            return {"status": "error", "message": f"플러시 실패: {str(e)}"}

    @app.get("/langfuse/traces", tags=["Langfuse"], response_model=Dict[str, Any])
    async def get_langfuse_traces(
        limit: int = 10,
        page: int = 1,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[str] = None,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Langfuse 트레이스 조회"""
        try:
            # 페이지 오프셋 계산
            offset = (page - 1) * limit
            
            # 트레이스 조회
            traces = rag_system.langfuse_manager.get_traces(
                limit=limit,
                offset=offset,
                user_id=user_id,
                session_id=session_id,
                name=name,
                tags=tags.split(',') if tags else None
            )
            
            return {
                "traces": traces,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "offset": offset
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"트레이스 조회 실패: {str(e)}")

    @app.get("/langfuse/traces/{trace_id}", tags=["Langfuse"])
    async def get_langfuse_trace_detail(trace_id: str, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """특정 트레이스 상세 조회"""
        try:
            trace = rag_system.langfuse_manager.get_trace(trace_id)
            if not trace:
                raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없습니다.")
            return trace
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"트레이스 조회 실패: {str(e)}")

    @app.get("/langfuse/observations", tags=["Langfuse"])
    async def get_langfuse_observations(
        limit: int = 10,
        page: int = 1,
        name: Optional[str] = None,
        type: Optional[str] = None,
        trace_id: Optional[str] = None,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Langfuse 관찰 데이터 조회"""
        try:
            offset = (page - 1) * limit
            
            observations = rag_system.langfuse_manager.get_observations(
                limit=limit,
                offset=offset,
                name=name,
                type=type,
                trace_id=trace_id
            )
            
            return {
                "observations": observations,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "offset": offset
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"관찰 데이터 조회 실패: {str(e)}")

    @app.get("/langfuse/sessions", tags=["Langfuse"])
    async def get_langfuse_sessions(
        limit: int = 10,
        page: int = 1,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Langfuse 세션 조회"""
        try:
            offset = (page - 1) * limit
            
            sessions = rag_system.langfuse_manager.get_sessions(
                limit=limit,
                offset=offset
            )
            
            return {
                "sessions": sessions,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "offset": offset
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"세션 조회 실패: {str(e)}")

    @app.get("/langfuse/usage", tags=["Langfuse"], response_model=LangfuseUsageResponse)
    async def get_langfuse_usage(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """Langfuse 사용량 통계"""
        try:
            usage_data = rag_system.langfuse_manager.get_usage_stats()
            return LangfuseUsageResponse(**usage_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"사용량 조회 실패: {str(e)}")

    @app.get("/langfuse/metrics", tags=["Langfuse"])
    async def get_langfuse_metrics(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = "day",
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Langfuse 메트릭 조회"""
        try:
            metrics = rag_system.langfuse_manager.get_metrics(
                start_date=start_date,
                end_date=end_date,
                granularity=granularity
            )
            return {"metrics": metrics}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"메트릭 조회 실패: {str(e)}")
