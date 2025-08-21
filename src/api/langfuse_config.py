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
    from langfuse.types import TraceContext
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None  # fallback for type hints
    TraceContext = None
    print("⚠️ Langfuse가 설치되지 않았습니다. pip install langfuse")

# Type checking imports
if TYPE_CHECKING:
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        CallbackHandler = None

logger = logging.getLogger(__name__)

class DebugCallbackHandler(CallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        print("[Langfuse DEBUG] on_llm_start called")
        return super().on_llm_start(*args, **kwargs)
    def on_llm_end(self, *args, **kwargs):
        print("[Langfuse DEBUG] on_llm_end called")
        return super().on_llm_end(*args, **kwargs)

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
            print("[Langfuse DEBUG] Langfuse가 설치되지 않았습니다.")
            return
        try:
            # 환경변수에서 설정 읽기
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
            print(f"[Langfuse DEBUG] ENV LANGFUSE_PUBLIC_KEY={public_key}")
            print(f"[Langfuse DEBUG] ENV LANGFUSE_SECRET_KEY={'SET' if secret_key else 'NOT SET'}")
            print(f"[Langfuse DEBUG] ENV LANGFUSE_HOST={host}")
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
                self.callback_handler = DebugCallbackHandler()
                print(f"[Langfuse DEBUG] CallbackHandler 생성: {self.callback_handler}")
                self.is_enabled = True
                logger.info(f"✅ Langfuse 초기화 완료: {host}")
                # 연결 테스트
                self._test_connection()
            else:
                logger.info("ℹ️ Langfuse 환경변수가 설정되지 않았습니다. Langfuse 추적이 비활성화됩니다.")
                print("[Langfuse DEBUG] Langfuse 환경변수 미설정. 콜백 생성 안됨.")
                self.is_enabled = False
        except Exception as e:
            logger.error(f"❌ Langfuse 초기화 실패: {str(e)}")
            print(f"[Langfuse DEBUG] Langfuse 초기화 예외: {str(e)}")
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
        print(f"[Langfuse DEBUG] get_callback_handler called. is_enabled={self.is_enabled}, callback_handler={self.callback_handler}")
        return self.callback_handler if self.is_enabled else None
    
    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """새로운 트레이스 생성"""
        if not self.is_enabled or not self.client:
            return None
        
        try:
            # trace_id 생성
            trace_id = self.client.create_trace_id()
            
            # TraceContext 생성 (딕셔너리 형태)
            if TraceContext:
                trace_context = TraceContext({'trace_id': trace_id})
                
                # span 시작 (올바른 파라미터 사용)
                span = self.client.start_span(
                    trace_context=trace_context,
                    name=name,
                    metadata=metadata or {}
                )
                return {
                    'trace_id': trace_id,
                    'span': span,
                    'trace_context': trace_context
                }
            else:
                return {
                    'trace_id': trace_id,
                    'span': None,
                    'trace_context': None
                }
        except Exception as e:
            logger.error(f"트레이스 생성 실패: {str(e)}")
            return None
    
    def log_generation(self, trace_context: Optional[Dict[str, Any]] = None, **kwargs):
        """Generation 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            # TraceContext 객체 생성
            tc = None
            if trace_context and TraceContext:
                if isinstance(trace_context, dict) and 'trace_id' in trace_context:
                    tc = TraceContext({'trace_id': trace_context['trace_id']})
                elif hasattr(trace_context, 'get') and trace_context.get('trace_id'):
                    tc = trace_context  # 이미 TraceContext 객체인 경우
            
            generation = self.client.start_generation(
                trace_context=tc,
                **kwargs
            )
            return generation
        except Exception as e:
            logger.error(f"Generation 로그 실패: {str(e)}")
    
    def log_span(self, trace_context: Optional[Dict[str, Any]] = None, name: str = "span", **kwargs):
        """Span 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            # TraceContext 객체 생성
            tc = None
            if trace_context and TraceContext:
                if isinstance(trace_context, dict) and 'trace_id' in trace_context:
                    tc = TraceContext({'trace_id': trace_context['trace_id']})
                elif hasattr(trace_context, 'get') and trace_context.get('trace_id'):
                    tc = trace_context  # 이미 TraceContext 객체인 경우
            
            span = self.client.start_span(
                trace_context=tc,
                name=name,
                **kwargs
            )
            return span
        except Exception as e:
            logger.error(f"Span 로그 실패: {str(e)}")
    
    def log_event(self, trace_context: Optional[Dict[str, Any]] = None, name: str = "event", **kwargs):
        """Event 로그"""
        if not self.is_enabled or not self.client:
            return
        
        try:
            # TraceContext 객체 생성
            tc = None
            if trace_context and TraceContext:
                if isinstance(trace_context, dict) and 'trace_id' in trace_context:
                    tc = TraceContext({'trace_id': trace_context['trace_id']})
                elif hasattr(trace_context, 'get') and trace_context.get('trace_id'):
                    tc = trace_context  # 이미 TraceContext 객체인 경우
            
            event = self.client.create_event(
                trace_context=tc,
                name=name,
                **kwargs
            )
            return event
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
    
    def get_traces(self, limit: int = 50, page: int = 1, **filters):
        """트레이스 목록 조회"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            # API를 통해 트레이스 조회
            result = self.client.api.trace.list(
                page=page,
                limit=limit,
                **filters
            )
            
            # Traces 객체를 딕셔너리로 변환
            return {
                "data": [trace.model_dump() if hasattr(trace, 'model_dump') else dict(trace) for trace in result.data],
                "meta": {
                    "page": result.meta.page,
                    "limit": result.meta.limit,
                    "total_items": result.meta.total_items,
                    "total_pages": result.meta.total_pages
                }
            }
        except Exception as e:
            logger.error(f"트레이스 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_trace_by_id(self, trace_id: str):
        """특정 트레이스 조회"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            result = self.client.api.trace.get(trace_id)
            
            # Trace 객체를 딕셔너리로 변환
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return dict(result)
        except Exception as e:
            logger.error(f"트레이스 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_observations(self, limit: int = 50, page: int = 1, **filters):
        """관찰 데이터 조회 (LLM 호출, 스팬 등)"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            result = self.client.api.observations.list(
                limit=limit,
                page=page,
                **filters
            )
            
            # Observations 객체를 딕셔너리로 변환
            return {
                "data": [obs.model_dump() if hasattr(obs, 'model_dump') else dict(obs) for obs in result.data],
                "meta": {
                    "page": result.meta.page,
                    "limit": result.meta.limit,
                    "total_items": result.meta.total_items,
                    "total_pages": result.meta.total_pages
                }
            }
        except Exception as e:
            logger.error(f"관찰 데이터 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_sessions(self, limit: int = 50, page: int = 1, **filters):
        """세션 목록 조회"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            result = self.client.api.sessions.list(
                limit=limit,
                page=page,
                **filters
            )
            
            # Sessions 객체를 딕셔너리로 변환
            return {
                "data": [session.model_dump() if hasattr(session, 'model_dump') else dict(session) for session in result.data],
                "meta": {
                    "page": result.meta.page,
                    "limit": result.meta.limit,
                    "total_items": result.meta.total_items,
                    "total_pages": result.meta.total_pages
                }
            }
        except Exception as e:
            logger.error(f"세션 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_metrics(self, **filters):
        """메트릭 데이터 조회"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            # metrics API가 존재하는지 확인
            if hasattr(self.client.api, 'metrics'):
                result = self.client.api.metrics.list(**filters)
                
                # Metrics 객체를 딕셔너리로 변환
                if hasattr(result, 'model_dump'):
                    return result.model_dump()
                else:
                    return dict(result)
            else:
                return {"error": "메트릭 API가 지원되지 않습니다"}
        except Exception as e:
            logger.error(f"메트릭 조회 실패: {str(e)}")
            return {"error": str(e)}
    
    def get_usage_statistics(self, from_date=None, to_date=None):
        """LLM 사용 통계 조회"""
        if not self.is_enabled or not self.client:
            return {"error": "Langfuse가 비활성화되었습니다"}
        
        try:
            filters = {}
            if from_date:
                filters['from_timestamp'] = from_date
            if to_date:
                filters['to_timestamp'] = to_date
            
            # 관찰 데이터에서 LLM 호출만 필터링
            observations = self.client.api.observations.list(
                type="GENERATION",
                limit=1000,
                **filters
            )
            
            # 사용 통계 계산
            total_tokens = 0
            total_cost = 0
            model_usage = {}
            
            if hasattr(observations, 'data'):
                for obs in observations.data:
                    if hasattr(obs, 'usage'):
                        if obs.usage:
                            total_tokens += getattr(obs.usage, 'total', 0)
                            total_cost += getattr(obs.usage, 'total_cost', 0)
                    
                    if hasattr(obs, 'model'):
                        model = obs.model or 'unknown'
                        if model not in model_usage:
                            model_usage[model] = {
                                'count': 0,
                                'tokens': 0,
                                'cost': 0
                            }
                        model_usage[model]['count'] += 1
                        if hasattr(obs, 'usage') and obs.usage:
                            model_usage[model]['tokens'] += getattr(obs.usage, 'total', 0)
                            model_usage[model]['cost'] += getattr(obs.usage, 'total_cost', 0)
            
            return {
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'model_usage': model_usage,
                'observation_count': len(observations.data) if hasattr(observations, 'data') else 0
            }
            
        except Exception as e:
            logger.error(f"사용 통계 조회 실패: {str(e)}")
            return {"error": str(e)}


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
