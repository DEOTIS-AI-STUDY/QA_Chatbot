"""
FastAPI용 Pydantic 모델 정의
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="질문 내용", min_length=1)
    session_id: str = Field(default="default", description="세션 ID")
    user_data: Optional[Dict[str, Any]] = Field(default=None, description="사용자 데이터")


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


# 파일 변환 관련 모델
class ConversionRequest(BaseModel):
    output_format: str = Field(..., description="출력 형식 (txt, json, pdf)")
    conversion_type: str = Field(default="default", description="변환 타입")


class ConversionResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    conversion_type: str
    processing_time: float


class SupportedFormatsResponse(BaseModel):
    supported_formats: List[str]
    conversion_types: Dict[str, List[str]]


# Admin 관련 모델들
class IndexChangeRequest(BaseModel):
    index_name: str = Field(..., description="변경할 인덱스 이름", min_length=1)


class IndexChangeResponse(BaseModel):
    status: str
    message: str
    current_index: str


class CurrentIndexResponse(BaseModel):
    current_index: str
    status: str


class IndexListResponse(BaseModel):
    indices: List[str]
    current_index: str
    count: int
    status: str


class IndexDetailedResponse(BaseModel):
    indices: List[Dict[str, Any]]
    current_index: str
    count: int
    status: str


# Ollama 모델 관리 관련 모델들
class OllamaModelInfo(BaseModel):
    name: str
    size: int
    digest: str
    modified_at: str
    details: Dict[str, Any]


class OllamaModelsResponse(BaseModel):
    models: List[OllamaModelInfo]
    count: int
    status: str
    message: str


class OllamaModelActionRequest(BaseModel):
    model_name: str = Field(..., description="Ollama 모델 이름 (예: llama3.1:8b)", min_length=1)


class OllamaModelActionResponse(BaseModel):
    success: bool
    message: str
    model_name: str


class OllamaStatusResponse(BaseModel):
    available: bool
    server_url: str
    message: str
    models_count: int
    models: List[OllamaModelInfo]
    status: str


class AvailableOllamaModelsResponse(BaseModel):
    available_models: List[str]
    count: int
    status: str
