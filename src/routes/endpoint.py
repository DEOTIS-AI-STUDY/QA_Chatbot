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
    ConversionResponse,
    IndexChangeRequest,
    IndexChangeResponse,
    CurrentIndexResponse,
    IndexListResponse,
    IndexDetailedResponse,
    OllamaModelInfo,
    OllamaModelsResponse,
    OllamaModelActionRequest,
    OllamaModelActionResponse,
    OllamaStatusResponse,
    AvailableOllamaModelsResponse
)

# 핵심 기능 모듈들
from config.app_config import FastAPIRAGSystem

# Langfuse 엔드포인트 import
from .endpoint_langfuse import create_langfuse_endpoints


def get_rag_system() -> FastAPIRAGSystem:
    """RAG 시스템 의존성 주입 함수"""
    # 이 함수는 main.py에서 덮어쓰여집니다.
    raise HTTPException(status_code=500, detail="RAG 시스템이 초기화되지 않았습니다.")


def create_endpoints(app, current_api_dir: str):
    """엔드포인트 생성 함수"""
    
    # Langfuse 엔드포인트도 함께 생성
    create_langfuse_endpoints(app)
    
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
        result = await rag_system.process_query_async(
            request.query, 
            request.session_id,
            request.user_data
        )
        
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

    # Admin 엔드포인트들 - Elasticsearch 인덱스 관리
    @app.post("/admin/change-index", response_model=IndexChangeResponse)
    async def change_elasticsearch_index(
        request: IndexChangeRequest,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Elasticsearch 인덱스를 동적으로 변경"""
        try:
            success = rag_system.change_elasticsearch_index(request.index_name)
            if success:
                return IndexChangeResponse(
                    status="success",
                    message=f"인덱스가 '{request.index_name}'로 변경되었습니다.",
                    current_index=rag_system.get_current_index()
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="인덱스 변경에 실패했습니다."
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"인덱스 변경 중 오류 발생: {str(e)}"
            )

    @app.get("/admin/current-index", response_model=CurrentIndexResponse)
    async def get_current_index(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """현재 사용 중인 Elasticsearch 인덱스 조회"""
        return CurrentIndexResponse(
            current_index=rag_system.get_current_index(),
            status="success"
        )

    @app.get("/admin/list-indices", response_model=IndexListResponse)
    async def list_available_indices(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """사용 가능한 Elasticsearch 인덱스 목록 조회"""
        try:
            indices = rag_system.list_available_indices()
            return IndexListResponse(
                indices=indices,
                current_index=rag_system.get_current_index(),
                count=len(indices),
                status="success"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"인덱스 목록 조회 중 오류 발생: {str(e)}"
            )

    @app.get("/admin/indices-detailed", response_model=IndexDetailedResponse)
    async def get_indices_detailed_info(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """Elasticsearch 인덱스 상세 정보 조회 (curl -s http://localhost:9200/_cat/indices?v 와 유사)"""
        try:
            indices_info = rag_system.get_indices_detailed_info()
            return IndexDetailedResponse(
                indices=indices_info,
                current_index=rag_system.get_current_index(),
                count=len(indices_info),
                status="success"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"인덱스 상세 정보 조회 중 오류 발생: {str(e)}"
            )

    # Ollama 모델 관리 엔드포인트들
    @app.get("/admin/ollama/status", response_model=OllamaStatusResponse)
    async def get_ollama_status(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """Ollama 서버 상태 및 설치된 모델 확인"""
        try:
            from core.models import ModelFactory
            
            available, message, status_info = ModelFactory.check_ollama_status()
            
            if available:
                models = [
                    OllamaModelInfo(
                        name=model['name'],
                        size=model['size'],
                        digest=model['digest'],
                        modified_at=model['modified_at'],
                        details=model['details']
                    ) for model in status_info.get('models', [])
                ]
                
                return OllamaStatusResponse(
                    available=True,
                    server_url=status_info.get('server_url', ''),
                    message=message,
                    models_count=status_info.get('models_count', 0),
                    models=models,
                    status="success"
                )
            else:
                return OllamaStatusResponse(
                    available=False,
                    server_url="",
                    message=message,
                    models_count=0,
                    models=[],
                    status="error"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama 상태 확인 중 오류 발생: {str(e)}"
            )

    @app.get("/admin/ollama/models", response_model=OllamaModelsResponse)
    async def list_ollama_models(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """설치된 Ollama 모델 목록 조회"""
        try:
            from core.models import ModelFactory
            
            models, message = ModelFactory.list_ollama_models()
            
            if models:
                model_objects = [
                    OllamaModelInfo(
                        name=model['name'],
                        size=model['size'],
                        digest=model['digest'],
                        modified_at=model['modified_at'],
                        details=model['details']
                    ) for model in models
                ]
                
                return OllamaModelsResponse(
                    models=model_objects,
                    count=len(model_objects),
                    status="success",
                    message=message
                )
            else:
                return OllamaModelsResponse(
                    models=[],
                    count=0,
                    status="error" if "실패" in message or "오류" in message else "success",
                    message=message
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama 모델 목록 조회 중 오류 발생: {str(e)}"
            )

    @app.get("/admin/ollama/available", response_model=AvailableOllamaModelsResponse)
    async def get_available_ollama_models(rag_system: FastAPIRAGSystem = Depends(get_rag_system)):
        """다운로드 가능한 Ollama 모델 목록 조회"""
        try:
            from core.models import ModelFactory
            
            available_models = ModelFactory.get_available_ollama_models_from_library()
            
            return AvailableOllamaModelsResponse(
                available_models=available_models,
                count=len(available_models),
                status="success"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"사용 가능한 모델 목록 조회 중 오류 발생: {str(e)}"
            )

    @app.post("/admin/ollama/add", response_model=OllamaModelActionResponse)
    async def add_ollama_model(
        request: OllamaModelActionRequest,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Ollama 모델 추가 (다운로드)"""
        try:
            from core.models import ModelFactory
            
            success, message = ModelFactory.add_ollama_model(request.model_name)
            
            return OllamaModelActionResponse(
                success=success,
                message=message,
                model_name=request.model_name
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"모델 추가 중 오류 발생: {str(e)}"
            )

    @app.delete("/admin/ollama/remove", response_model=OllamaModelActionResponse)
    async def remove_ollama_model(
        request: OllamaModelActionRequest,
        rag_system: FastAPIRAGSystem = Depends(get_rag_system)
    ):
        """Ollama 모델 삭제"""
        try:
            from core.models import ModelFactory
            
            success, message = ModelFactory.remove_ollama_model(request.model_name)
            
            return OllamaModelActionResponse(
                success=success,
                message=message,
                model_name=request.model_name
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"모델 삭제 중 오류 발생: {str(e)}"
            )

    # 파일 변환 관련 엔드포인트들
    @app.get("/converter/formats")
    async def get_supported_formats():
        """지원하는 변환 형식 목록"""
        converter = FileConverterFactory()
        return {
            "supported_formats": converter.get_supported_formats(),
            "converters": {
                "txt": converter.get_conversion_types("txt"),
                "json": converter.get_conversion_types("json"),
                "pdf": converter.get_conversion_types("pdf")
            }
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
