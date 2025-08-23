#!/usr/bin/env python3
"""
FastAPI 엔드포인트 정의
- RAG 시스템 의존성 주입을 통한 초기화 문제 해결
- 모든 API 엔드포인트를 여기에 정의
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import HTTPException, Request, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse

# 파일 변환 모듈 import  
from converters.factory import FileConverterFactory

# Pydantic 모델들
from models.api_models import (
    QueryRequest, 
    InitRequest, 
    ChatHistoryResponse,
    LangfuseUsageResponse,
    ConversionResponse
)

# 핵심 기능 모듈들
from config.app_config import FastAPIRAGSystem


def get_rag_system() -> FastAPIRAGSystem:
    """RAG 시스템 의존성 주입 함수"""
    # 이 함수는 main.py에서 덮어쓰여집니다.
    raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")


def create_endpoints(app, current_api_dir: str):
    """엔드포인트 생성 함수"""
    
    # API 엔드포인트들
    @app.get("/")
    async def read_root():
        """루트 엔드포인트 - API 기본 정보"""
        return {
            "message": "FastAPI RAG 시스템이 실행 중입니다",
            "docs": "/docs",
            "status": "ok"
        }

    @app.get("/converter/test")
    async def converter_test():
        """변환기 테스트 페이지"""
        return FileResponse(os.path.join(current_api_dir, "converter_test.html"))

    @app.get("/health")
    async def health_check():
        """헬스 체크"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/dependencies")
    async def check_dependencies(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """의존성 확인"""
        return await rag_system.check_dependencies_async()

    @app.get("/models")
    async def get_available_models(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """사용 가능한 모델 목록"""
        return {
            "models": rag_system.model_factory.get_available_models(),
            "status": "success"
        }

    @app.post("/initialize")
    async def initialize_system(request: InitRequest, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
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
    async def get_system_status(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """시스템 상태 확인"""
        return rag_system.get_system_info()

    @app.post("/query")
    async def process_query(request: QueryRequest, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """질의 처리"""
        result = await rag_system.process_query_async(request.query, request.session_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result

    @app.get("/chat/history/{session_id}")
    async def get_chat_history(session_id: str, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """대화 기록 조회"""
        chat_manager = rag_system.get_chat_manager(session_id)
        history = chat_manager.chat_history
        
        return ChatHistoryResponse(
            session_id=session_id,
            history=history,
            count=len(history)
        )

    @app.delete("/chat/history/{session_id}")
    async def clear_chat_history(session_id: str, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """대화 기록 삭제"""
        if session_id in rag_system.session_managers:
            rag_system.session_managers[session_id].clear_history()
            return {"status": "success", "message": f"세션 {session_id}의 대화 기록이 삭제되었습니다."}
        else:
            return {"status": "info", "message": f"세션 {session_id}가 존재하지 않습니다."}

    @app.get("/sessions")
    async def get_active_sessions(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """활성 세션 목록"""
        return {
            "active_sessions": list(rag_system.session_managers.keys()),
            "count": len(rag_system.session_managers)
        }

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """세션 삭제"""
        if session_id in rag_system.session_managers:
            del rag_system.session_managers[session_id]
            return {"status": "success", "message": f"세션 {session_id}가 삭제되었습니다."}
        else:
            return {"status": "info", "message": f"세션 {session_id}가 존재하지 않습니다."}

    # Langfuse 관련 엔드포인트들
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

    # 파일 변환 관련 엔드포인트들
    @app.get("/converter/formats")
    async def get_supported_formats():
        """지원하는 변환 형식 목록"""
        converter = FileConverterFactory()
        return {
            "supported_formats": ["txt", "json", "pdf"],
            "converters": converter.get_available_converters()
        }

    @app.post("/convert/file", response_model=ConversionResponse)
    async def convert_file(
        file: UploadFile = File(...),
        output_format: str = Form(...),
        conversion_type: str = Form(default="default")
    ):
        """업로드된 파일 변환"""
        start_time = time.time()
        
        try:
            # 파일 내용 읽기
            file_content = await file.read()
            
            # 파일 확장자 확인
            file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            
            if not file_extension:
                raise HTTPException(status_code=400, detail="파일 확장자를 확인할 수 없습니다.")
            
            # 변환기 생성
            converter_factory = FileConverterFactory()
            converter = converter_factory.create_converter(output_format)
            
            if not converter:
                raise HTTPException(
                    status_code=400, 
                    detail=f"지원하지 않는 출력 형식입니다: {output_format}"
                )
            
            # 변환 실행
            if hasattr(converter, 'convert_from_bytes'):
                converted_content = converter.convert_from_bytes(
                    file_content, 
                    file.filename,
                    conversion_type
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="변환기가 바이트 변환을 지원하지 않습니다."
                )
            
            # 안전한 파일명 생성
            safe_base_filename = "".join(c for c in file.filename.rsplit('.', 1)[0] if c.isalnum() or c in (' ', '-', '_')).strip()
            
            # 출력 파일명 결정
            if output_format == "pdf":
                file_extension = "pdf"
            elif output_format == "json":
                file_extension = "json"
            else:  # txt
                file_extension = "txt"
            
            if conversion_type == "default":
                output_filename = f"{safe_base_filename}.{file_extension}"
            else:
                output_filename = f"{safe_base_filename}_{conversion_type}.{file_extension}"
            
            processing_time = time.time() - start_time
            
            return ConversionResponse(
                status="success",
                message="파일 변환이 성공적으로 완료되었습니다.",
                filename=output_filename,
                file_size=len(converted_content),
                conversion_type=conversion_type,
                processing_time=processing_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            return ConversionResponse(
                status="error",
                message=f"파일 변환 실패: {str(e)}",
                conversion_type=conversion_type,
                processing_time=processing_time
            )
