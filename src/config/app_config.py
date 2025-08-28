"""
환경 설정 및 애플리케이션 초기화 로직
"""

import json
import os
import sys
from typing import Dict, Any
from contextlib import asynccontextmanager

from core.config import LLM_MODELS, HUGGINGFACE_EMBEDDINGS_AVAILABLE, OLLAMA_AVAILABLE
from core.models import ModelFactory
from core.rag import create_llm_chain, create_rag_chain, create_retriever, create_enhanced_retriever, prompt_for_refined_query, prompt_for_query, prompt_for_context_summary
from core.chat_history import ChatHistoryManager
from utils.elasticsearch import ElasticsearchManager


def load_environment():
    """환경 변수 로드"""
    try:
        from dotenv import load_dotenv
        # 프로젝트 루트에서 .env.prod 파일 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        env_path = os.path.join(root_dir, '.env.prod')
        load_dotenv(env_path)
        print(f"🔧 환경 변수 로드: {env_path}")
        print(f"🔗 OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'Not set')}")
        
        # LangSmith 트레이싱 명시적 비활성화
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if "LANGCHAIN_API_KEY" in os.environ:
            del os.environ["LANGCHAIN_API_KEY"]
        if "LANGSMITH_API_KEY" in os.environ:
            del os.environ["LANGSMITH_API_KEY"]
        print("🚫 LangSmith 트레이싱 비활성화됨")
        
    except ImportError:
        print("⚠️ python-dotenv가 설치되지 않았습니다. 환경 변수를 수동으로 설정해주세요.")


def setup_paths():
    """프로젝트 경로 설정"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)


def get_langfuse_config():
    """Langfuse 설정 가져오기"""
    try:
        # Docker에서 모듈로 실행할 때
        from api.langfuse_config import get_langfuse_manager, get_langfuse_callback
    except ImportError:
        try:
            # 로컬에서 직접 실행할 때
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
            from langfuse_config import get_langfuse_manager, get_langfuse_callback
        except ImportError:
            print("⚠️ Langfuse 설정을 불러올 수 없습니다.")
            return None, None
    
    return get_langfuse_manager, get_langfuse_callback


def check_elasticsearch_availability():
    """Elasticsearch 가용성 확인"""
    try:
        from elasticsearch import Elasticsearch
        return True
    except ImportError:
        print("⚠️ Elasticsearch가 설치되지 않았습니다. pip install elasticsearch")
        return False


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
        
        # 미리 생성된 체인들 (최적화)
        self.refinement_chain = None
        self.qa_chain = None
        self.summary_chain = None
        
        # Langfuse 매니저 초기화
        get_langfuse_manager, _ = get_langfuse_config()
        self.langfuse_manager = get_langfuse_manager() if get_langfuse_manager else None
        
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
        import asyncio
        
        def _check_dependencies():
            issues = []
            
            if not check_elasticsearch_availability():
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
        import asyncio
        import time
        
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
                _, get_langfuse_callback = get_langfuse_config()
                langfuse_callback = get_langfuse_callback() if get_langfuse_callback else None
                callbacks = [langfuse_callback] if langfuse_callback else None
                
                self.rag_chain, success_or_error = create_rag_chain(
                    embeddings=self.embedding_model,
                    llm_model=self.llm_model,
                    top_k=top_k,
                    callbacks=callbacks
                )

                # 고도화된 하이브리드 Retriever 생성
                self.retriever = create_enhanced_retriever(
                    embedding_model=self.embedding_model,
                    top_k=top_k
                )

                # 미리 사용할 체인들 생성 (최적화)
                self.refinement_chain = create_llm_chain(
                    self.llm_model, 
                    prompt_for_refined_query,
                    input_variables=["userinfo", "question", "context"]
                )
                
                self.qa_chain = create_llm_chain(
                    self.llm_model,
                    prompt_for_query,
                    input_variables=["question", "context"]
                )
                
                self.summary_chain = create_llm_chain(
                    self.llm_model,
                    prompt_for_context_summary,
                    input_variables=["context"]
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
    
    def enhance_docs_for_table_preservation(self, merged_docs):
        """표 구조를 보존하고 LLM의 이해를 돕는 문서 결합 함수"""
        enhanced_docs = []
        
        for i, doc in enumerate(merged_docs):
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {})
            
            # 표 포함 여부 확인 (마크다운 표 패턴)
            has_table = bool(
                "|" in content and 
                ("---" in content or ":-:" in content or ":--" in content or "--:" in content)
            )
            
            if has_table:
                # 표가 포함된 문서는 특별한 마킹과 함께 보존
                enhanced_content = f"""📊 **표 데이터 문서 #{i+1}** (파일: {metadata.get('filename', 'Unknown')})

    {content.strip()}

    📊 **표 데이터 문서 끝**"""
            else:
                # 일반 문서
                enhanced_content = f"""📄 **참고 문서 #{i+1}**
    (파일: {metadata.get('filename', 'Unknown')})

    {content.strip()}

    📄 **문서 끝**"""
            
            enhanced_docs.append(enhanced_content)
        
        # 각 문서 사이에 명확한 구분자를 넣어 결합
        separator = "\n\n" + "="*50 + " 문서 구분선 " + "="*50 + "\n\n"
        return separator.join(enhanced_docs)



    async def process_query_async(self, query: str, session_id: str = "default", user_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """질의 처리 (비동기 버전, 의미+키워드 검색 병합)"""
        if not self.is_initialized or not self.rag_chain:
            return {
                "status": "error",
                "message": "RAG 시스템이 초기화되지 않았습니다."
            }

        import asyncio
        import time

        def _process_query():
            try:
                start_time = time.time()

                # Langfuse 트레이스 생성
                trace = None
                if self.langfuse_manager:
                    trace_metadata = {
                        "session_id": session_id,
                        "model": self.model_choice,
                        "top_k": self.top_k
                    }
                    # 사용자 데이터가 있으면 트레이스 메타데이터에 추가
                    if user_data and isinstance(user_data, dict):
                        trace_metadata.update({
                            "user_id": user_data.get("userId"),
                            "user_name": user_data.get("userName"),
                            "user_age": user_data.get("age"),
                            "user_income": user_data.get("income"),
                            "is_authenticated": user_data.get("isAuthenticated"),
                            "login_time": user_data.get("loginTime")
                        })
                    
                    trace = self.langfuse_manager.create_trace(
                        name="rag_query",
                        metadata=trace_metadata
                    )

                # 대화 기록 관리자 가져오기
                chat_manager = self.get_chat_manager(session_id)

                # 대화 기록으로 질문 재정의
                history = chat_manager.build_history()
                print(f"🔍 대화 기록: {history}")
                print(f"🔍 원본 질의: {query}")

                # # 질문 재정의를 위한 초기 검색 (의미 + 키워드)
                # initial_semantic_docs = self.retriever.get_relevant_documents(query)
                # initial_keyword_results = ElasticsearchManager.keyword_search(query, top_k=3)
                
                # # 초기 검색 결과 병합 (재질의용)
                # initial_context = []
                # for doc in initial_semantic_docs[:3]:  # 상위 3개만
                #     initial_context.append(getattr(doc, "page_content", str(doc)))
                # for kdoc in initial_keyword_results[:2]:  # 상위 2개만
                #     content = kdoc.get("content", "")
                #     if content and content not in initial_context:
                #         initial_context.append(content)

                # 질문을 알맞게 변경하기위함이기에 history만을 context에 사용
                userinfo = {  
                    "userId": "bccard",  
                    "userName": "김명정",  
                    "loginTime": "2025-08-27T14:23:45.123Z",
                    "isAuthenticated": True, # python에서 true -> True 로 치환됨
                    "age": "27",
                    "income": "77,511,577",
                    "data": {
                        "email": "kmj@deotis.co.kr",
                        "phone": "010-1234-5678",
                        "ownCardArr": [
                            {
                                "bank": "우리카드",
                                "name": "VVIP 카드",
                                "paymentDate": "4",
                                "ipn": "VISA",
                                "type": "신용카드"
                            }
                        ]
                    }
                }  # JSON 객체 형태의 사용자 정보
                
                refined_query_str = self.refinement_chain.run({"question": query, "context": history, "userinfo": userinfo})
                print(f"🔍 정제된 질의 (원본): {refined_query_str}")
                
                # JSON 또는 마크다운 형태 처리
                refined_query = query  # 기본값
                action = None
                
                def extract_json_from_markdown(text):
                    """마크다운에서 JSON 블록 추출"""
                    import re
                    # ```json ... ``` 또는 ```  ... ``` 형태의 코드 블록 찾기
                    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
                    return None
                
                def parse_refined_query(response_str):
                    """JSON 또는 마크다운 응답을 파싱하여 refined_query 추출"""
                    # 1. 직접 JSON 파싱 시도
                    try:
                        parsed_json = json.loads(response_str)
                        return parsed_json.get('refined_query'), parsed_json.get('action')
                    except json.JSONDecodeError:
                        pass
                    
                    # 2. 마크다운에서 JSON 추출 시도
                    json_content = extract_json_from_markdown(response_str)
                    if json_content:
                        try:
                            parsed_json = json.loads(json_content)
                            return parsed_json.get('refined_query'), parsed_json.get('action')
                        except json.JSONDecodeError:
                            pass
                    
                    # 3. 텍스트에서 직접 추출 시도 (마크다운 텍스트 내용)
                    # refined_query: 또는 질문: 등의 패턴 찾기
                    import re
                    query_patterns = [
                        r'refined_query[:\s]*([^\n]+)',
                        r'질문[:\s]*([^\n]+)',
                        r'정제된\s*질의[:\s]*([^\n]+)',
                        r'개선된\s*질문[:\s]*([^\n]+)'
                    ]
                    
                    for pattern in query_patterns:
                        match = re.search(pattern, response_str, re.IGNORECASE)
                        if match:
                            extracted_query = match.group(1).strip()
                            # 따옴표 제거
                            extracted_query = extracted_query.strip('"\'')
                            return extracted_query, None
                    
                    # 4. action 추출 시도
                    action_patterns = [
                        r'action[:\s]*([^\n]+)',
                        r'동작[:\s]*([^\n]+)'
                    ]
                    
                    extracted_action = None
                    for pattern in action_patterns:
                        match = re.search(pattern, response_str, re.IGNORECASE)
                        if match:
                            extracted_action = match.group(1).strip().strip('"\'')
                            break
                    
                    return None, extracted_action
                
                try:
                    refined_query, action = parse_refined_query(refined_query_str)
                    
                    if refined_query:
                        print(f"🔍 정제된 질의: {refined_query}")
                    else:
                        print(f"🔍 정제된 질의 추출 실패, 원본 질의 사용: {query}")
                        refined_query = query
                    
                    if action == 'reset':
                        # 대화 기록 초기화
                        chat_manager.clear_history()
                        print("🔄 대화 기록 초기화됨")
                    elif action == 'answer':
                        # refined_query를 답변으로 사용하고 바로 리턴
                        processing_time = time.time() - start_time
                        print(f"🔍 직접 답변 모드: {refined_query}")
                        
                        # 대화 기록에 질문과 답변 추가
                        chat_manager.add_chat(query, refined_query)
                        
                        # Langfuse에 결과 로그
                        if trace and self.langfuse_manager:
                            self.langfuse_manager.log_generation(
                                trace_context=trace.get('trace_context'),
                                name="direct_answer_generation",
                                input=query,
                                output=refined_query,
                                metadata={
                                    "processing_time": processing_time,
                                    "mode": "direct_answer",
                                    "model": self.model_choice
                                }
                            )
                        
                        return {
                            "status": "success",
                            "answer": refined_query,
                            "query": query,
                            "refined_query": refined_query,
                            "session_id": session_id,
                            "processing_time": processing_time,
                            "retrieved_docs": []
                        }
                        
                except Exception as e:
                    print(f"❌ 정제된 질의 파싱 실패: {str(e)}")
                    print(f"❌ 응답 내용: {refined_query_str}")
                    refined_query = query
                    chat_manager.clear_history()
                    print("🔄 대화 기록 초기화됨 (파싱 실패)")

                # 고도화된 하이브리드 검색 (시맨틱 + 키워드 + 스코어링이 모두 포함됨)
                merged_docs = self.retriever.get_relevant_documents(refined_query)
                print(f"🔍 고도화된 하이브리드 검색 결과 개수: {len(merged_docs)}")

                # # 문서를 텍스트로 변환
                docs_text = "\n\n---\n\n".join([
                    getattr(doc, "page_content", str(doc)) for doc in merged_docs
                ])
                print(f"🔍 최종 문서 컨텍스트 길이: {len(docs_text)} 문자")
                # docs_text = self.enhance_docs_for_table_preservation(merged_docs)
                # print(f"🔍 최종 문서 컨텍스트 길이: {len(docs_text)} 문자")


                # 개인 정보 관련 질의인지 판단 // 판단만 하고 아직 쓰진 않음
                personal_keywords = ['내', '나의', '내가', '내 카드', '내 정보', '내 결제일', '내 혜택', '내 포인트']
                is_personal_query = any(keyword in query for keyword in personal_keywords)
    

                # 사용자 정보 기반 개인화된 컨텍스트 생성
                personalized_context = docs_text
                if user_data and isinstance(user_data, dict):
                    user_context = f"""
사용자 정보:
- 이름: {user_data.get('userName', 'N/A')}
- 나이: {user_data.get('age', 'N/A')}세
- 연소득: {user_data.get('income', 'N/A')}원
- 인증 상태: {'인증됨' if user_data.get('isAuthenticated') else '미인증'}
"""
                    # 보유 카드 정보 추가
                    user_data_obj = user_data.get('data', {})
                    if user_data_obj and isinstance(user_data_obj, dict) and user_data_obj.get('ownCardArr'):
                        user_context += "\n보유 카드:\n"
                        for card in user_data_obj['ownCardArr']:
                            if isinstance(card, dict):
                                user_context += f"- {card.get('bank', 'N/A')} {card.get('name', 'N/A')} ({card.get('type', 'N/A')}, 결제일: {card.get('paymentDate', 'N/A')}일)\n"
                    
                    personalized_context = user_context + "\n\n관련 문서:\n" + docs_text
                    print(f"🔍 개인화된 컨텍스트 길이: {len(personalized_context)} 문자")

                # 검색된 자료와 재정의 질문을 LLM에 넘겨서 답변 생성
                result = self.qa_chain.invoke({"question": refined_query, "context": personalized_context})

                # 디버깅: 실제 응답 구조 출력
                print(f"🔍 RAG 체인 응답 구조: {result}")
                print(f"🔍 응답 키들: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

                processing_time = time.time() - start_time

                # RetrievalQA는 'result' 키를 사용함
                if result and ('answer' in result or 'result' in result or 'text' in result):
                    answer = result.get('answer') or result.get('result') or result.get('text')
                    print(f"🔍 최종 답변: {answer}")
                    # 답변 요약
                    # answer_summary = self.summary_chain.run({"context": answer})
                    # print(f"🔍 답변 요약: {answer_summary}")
                    # 대화 기록에 질문과 답변 추가
                    chat_manager.add_chat(query, answer)

                    # Langfuse에 결과 로그
                    if trace and self.langfuse_manager:
                        self.langfuse_manager.log_generation(
                            trace_context=trace.get('trace_context'),
                            name="rag_generation",
                            input=query,
                            output=answer,
                            metadata={
                                "processing_time": processing_time,
                                "retrieved_docs_count": len(merged_docs),
                                "model": self.model_choice
                            }
                        )

                    # retrieved_docs에 고도화된 검색 결과 포함
                    retrieved_docs = []
                    for doc in merged_docs:
                        retrieved_docs.append({
                            "type": "enhanced_hybrid",
                            "content": getattr(doc, "page_content", str(doc)),
                            "metadata": getattr(doc, "metadata", {})
                        })

                    return {
                        "status": "success",
                        "answer": answer,
                        "query": query,
                        "refined_query": refined_query,
                        "session_id": session_id,
                        "username": user_data.get("userName", "") if user_data and isinstance(user_data, dict) else "",  # 사용자 데이터 포함
                        "processing_time": processing_time,
                        "retrieved_docs": retrieved_docs
                    }
                else:
                    # Langfuse에 에러 로그
                    if trace and self.langfuse_manager:
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
                if 'trace' in locals() and trace and self.langfuse_manager:
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
            "langfuse_status": self.langfuse_manager.get_status() if self.langfuse_manager else {"available": False}
        }


@asynccontextmanager
async def lifespan(app):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시
    global rag_system
    rag_system = FastAPIRAGSystem()
    print("🚀 FastAPI RAG 시스템 시작")
    
    yield
    
    # 종료 시
    print("👋 FastAPI RAG 시스템 종료")


# 전역 RAG 시스템 인스턴스
rag_system = None


# Langfuse 함수들을 export
def get_langfuse_manager():
    """Langfuse 매니저 가져오기"""
    try:
        get_langfuse_manager_func, _ = get_langfuse_config()
        return get_langfuse_manager_func() if get_langfuse_manager_func else None
    except Exception:
        return None


def get_langfuse_callback():
    """Langfuse 콜백 가져오기"""
    try:
        _, get_langfuse_callback_func = get_langfuse_config()
        return get_langfuse_callback_func() if get_langfuse_callback_func else None
    except Exception:
        return None
