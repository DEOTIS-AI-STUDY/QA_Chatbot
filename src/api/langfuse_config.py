#!/usr/bin/env python3
"""
Langfuse 설정 및 초기화 모듈
- Langfuse 클라이언트 초기화
- FastAPI와 Langchain 통합
- 추적 및 모니터링 설정
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from contextlib import asynccontextmanager

try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None  # fallback for type hints
    print("⚠️ Langfuse가 설치되지 않았습니다. pip install langfuse")

# Type checking imports
if TYPE_CHECKING:
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        CallbackHandler = None

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Langfuse 관리 클래스"""
    
    def __init__(self):
        self.client = None  # Optional[Langfuse]
        self.callback_handler = None  # Optional[CallbackHandler]
        self.is_enabled = False
        self._initialize()
    
    def _initialize(self):
        """Langfuse 클라이언트 초기화"""
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse가 설치되지 않았습니다.")
            return
        
        try:
            # 환경변수에서 설정 읽기
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if public_key and secret_key:
                # 환경변수 설정 (CallbackHandler가 자동으로 읽음)
                os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
                os.environ["LANGFUSE_SECRET_KEY"] = secret_key
                os.environ["LANGFUSE_HOST"] = host
                
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                
                # CallbackHandler 초기화 (환경변수에서 자동으로 읽음)
                self.callback_handler = CallbackHandler()
                
                self.is_enabled = True
                logger.info(f"✅ Langfuse 초기화 완료: {host}")
                
                # 연결 테스트
                self._test_connection()
                
            else:
                logger.info("ℹ️ Langfuse 환경변수가 설정되지 않았습니다. Langfuse 추적이 비활성화됩니다.")
                self.is_enabled = False
                
        except Exception as e:
            logger.error(f"❌ Langfuse 초기화 실패: {str(e)}")
            self.is_enabled = False
    
    def _test_connection(self):
        """Langfuse 연결 테스트"""
        try:
            if self.client:
                # 연결 테스트 - auth_check 사용
                result = self.client.auth_check()
                if result:
                    logger.info("✅ Langfuse 연결 테스트 성공")
                else:
                    logger.warning("⚠️ Langfuse 인증 실패")
        except Exception as e:
            logger.warning(f"⚠️ Langfuse 연결 테스트 실패: {str(e)}")
    
    def get_callback_handler(self):
        """LangChain 콜백 핸들러 반환"""
        return self.callback_handler if self.is_enabled else None
    
    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """새로운 트레이스 생성"""
        if not self.is_enabled or not self.client:
            return None
        
        try:
            # trace_id 생성 후 span 시작
            trace_id = self.client.create_trace_id()
            span = self.client.start_span(
                name=name,
                trace_id=trace_id,
                metadata=metadata or {}
            )
            return span
        except Exception as e:
            logger.error(f"트레이스 생성 실패: {str(e)}")
            return None
    
    def log_generation(self, trace_id: str, **kwargs):
        """Generation 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.start_generation(**kwargs)
        except Exception as e:
            logger.error(f"Generation 로그 실패: {str(e)}")
    
    def log_span(self, trace_id: str, name: str, **kwargs):
        """Span 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.start_span(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Span 로그 실패: {str(e)}")
    
    def log_event(self, trace_id: str, name: str, **kwargs):
        """Event 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            self.client.create_event(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Event 로그 실패: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Langfuse 상태 반환"""
        return {
            "available": LANGFUSE_AVAILABLE,
            "enabled": self.is_enabled,
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "public_key_set": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
            "secret_key_set": bool(os.getenv("LANGFUSE_SECRET_KEY"))
        }
    
    def flush(self):
        """대기 중인 로그를 강제로 전송"""
        if self.is_enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Flush 실패: {str(e)}")


# 전역 Langfuse 매니저 인스턴스
langfuse_manager = LangfuseManager()


def get_langfuse_manager() -> LangfuseManager:
    """Langfuse 매니저 인스턴스 반환"""
    return langfuse_manager


def get_langfuse_callback():
    """LangChain용 Langfuse 콜백 반환"""
    return langfuse_manager.get_callback_handler()


# FastAPI용 리이프스팬 이벤트
@asynccontextmanager
async def langfuse_lifespan():
    """Langfuse 라이프사이클 관리"""
    logger.info("🚀 Langfuse 서비스 시작")
    yield
    # 종료 시 대기 중인 로그 전송
    langfuse_manager.flush()
    logger.info("👋 Langfuse 서비스 종료")
