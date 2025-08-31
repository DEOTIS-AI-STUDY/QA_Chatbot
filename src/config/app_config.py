"""
환경 설정 및 애플리케이션 초기화 로직
"""

import json
import os
import sys
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

from core.config import LLM_MODELS, HUGGINGFACE_EMBEDDINGS_AVAILABLE, OLLAMA_AVAILABLE
from core.models import ModelFactory
from core.rag import create_llm_chain, create_rag_chain, create_retriever, create_enhanced_retriever, prompt_for_refined_query, prompt_for_query, prompt_for_context_summary
from core.chat_history import ChatHistoryManager
from utils.elasticsearch import ElasticsearchManager


def add_related_links(answer_text, question_text=""):
    """
    답변에 관련 링크를 자동으로 추가하는 함수
    """
    # 키워드-링크 매핑
    keyword_links = {
        '카드발급': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0792',
        '이용한도': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind1113',
        '결제일': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0618',
        '이용기간': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0623',
        '리볼빙': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind1187',
        '교통카드': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0649',
        '신용카드': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0667',
        '포인트': 'https://isson.bccard.com/3rd/openSigninFormPage.jsp?UURL=https%3A%2F%2Fisson.bccard.com%2Fnls3%2Ffcs&NONCE=tvaQoSYB9J90I5r1z%2Bu2gNqawETc7ThhYPlG%2Fz308%2FoRCuqBsL%2F6dQjzXnAfZ2CjYEisW42xcJTSYKyTiQfcwQ%3D%3D&FORM=777',
        '혜택': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind1200',
        '대출': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0667',
        '할부': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0667',
        '연체': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0671',
        '소득공제': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0670',
        '해외': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0650',
        '장애': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0791',
        '분실': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind0901',
        '부가서비스': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind1114',
        '연체 절차': 'https://www.bccard.com/app/card/ContentsLinkActn.do?pgm_id=ind1115'
    }
    
    # 이미 링크가 있는지 확인
    if '---' in answer_text and '자세한 사항을' in answer_text:
        return answer_text
    
    # 답변과 질문에서 키워드 찾기
    found_keywords = []
    search_text = (answer_text + " " + question_text).lower()
    
    for keyword, link in keyword_links.items():
        if keyword in search_text:
            found_keywords.append((keyword, link))
    
    # 최대 3개까지만 선택 (질문 우선순위)
    if found_keywords:
        # 질문에 있는 키워드 우선
        question_keywords = []
        answer_keywords = []
        
        for keyword, link in found_keywords:
            if keyword in question_text.lower():
                question_keywords.append((keyword, link))
            else:
                answer_keywords.append((keyword, link))
        
        # 질문 키워드 + 답변 키워드 조합해서 최대 3개
        selected_keywords = (question_keywords + answer_keywords)[:3]
        
        if selected_keywords:
            link_section = "\n\n---\n자세한 사항을 알고 싶으시면 아래 링크를 참고하세요:\n"
            for keyword, link in selected_keywords:
                link_section += f"[{keyword}]({link})\n"
            
            return answer_text + link_section
    
    return answer_text


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
        # 핵심 컴포넌트
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
        
        # 외부 서비스 매니저
        get_langfuse_manager, _ = get_langfuse_config()
        self.langfuse_manager = get_langfuse_manager() if get_langfuse_manager else None
        
        # 상태 관리
        self.session_managers = {}
        self.is_initialized = False
        self.initialization_time = None

        # 인덱스 관리
        self.current_index_name = os.getenv("INDEX_NAME", "yang_deotis_rag")
    
    # =============================================================================
    # 세션 및 상태 관리 메서드
    # =============================================================================
    
    def get_chat_manager(self, session_id: str = "default") -> ChatHistoryManager:
        """세션별 대화 기록 관리자 반환"""
        if session_id not in self.session_managers:
            self.session_managers[session_id] = ChatHistoryManager(max_history=10)
        return self.session_managers[session_id]
    
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
    
    # =============================================================================
    # 시스템 초기화 및 의존성 확인
    # =============================================================================
    
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
    
    # =============================================================================
    # 문서 처리 및 검색 유틸리티
    # =============================================================================
    
    def convert_docs_with_table_preservation(self, merged_docs):
        """표 구조를 보존하는 고급 문서 변환"""
        enhanced_sections = []
        
        for i, doc in enumerate(merged_docs):
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {})
            
            # 문서 헤더 정보
            doc_header = f"""
    📄 **문서 {i+1}**: {metadata.get('filename', '알 수 없는 파일')}
    📊 **페이지**: {metadata.get('page_number', 'N/A')}
    🏷️ **카테고리**: {metadata.get('category', '일반')}
    ⭐ **관련성 점수**: {getattr(doc, '_score', 'N/A')}
    """
            
            # 표 포함 여부 확인 및 처리
            if self._has_table_structure(content):
                processed_content = self._preserve_table_structure(content)
                doc_section = f"""{doc_header}
    📊 **[표 데이터 포함 문서]**

    {processed_content}

    📊 **[표 데이터 끝]**
    """
            else:
                doc_section = f"""{doc_header}
    📝 **[일반 텍스트 문서]**

    {content.strip()}

    📝 **[문서 끝]**
    """
            
            enhanced_sections.append(doc_section)
        
        return "\n\n" + "="*80 + "\n\n".join(enhanced_sections)

    def _has_table_structure(self, content):
        """표 구조 포함 여부 확인"""
        table_indicators = [
            "|" in content and ("---" in content or ":-:" in content),
            content.count("\t") > 5,  # 탭으로 구분된 표
            re.search(r'\d+\.\s+.*?\s+\d+', content),  # 번호 + 텍스트 + 숫자 패턴
        ]
        return any(table_indicators)

    def _preserve_table_structure(self, content):
        """표 구조 보존 처리"""
        # 마크다운 표 형태로 변환
        if "|" in content:
            return self._format_markdown_table(content)
        
        # 탭 구분 표를 마크다운으로 변환
        if "\t" in content:
            return self._convert_tab_to_markdown(content)
        
        # 일반 텍스트를 구조화된 형태로 변환
        return self._structure_plain_text_table(content)

    def _format_markdown_table(self, content):
        """마크다운 표 형식 정리"""
        lines = content.split('\n')
        table_lines = []
        
        for line in lines:
            if '|' in line:
                # 표 라인 정리 및 포맷팅
                cells = [cell.strip() for cell in line.split('|')]
                formatted_line = "| " + " | ".join(cells) + " |"
                table_lines.append(formatted_line)
            elif line.strip():
                table_lines.append(line)
        
        return '\n'.join(table_lines)

    # =============================================================================
    # 질의 분석 및 처리 유틸리티
    # =============================================================================
    
    def _extract_json_from_markdown(self, text):
        """마크다운에서 JSON 블록 추출"""
        import re
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _parse_query_analysis(self, response_str):
        """새로운 JSON 응답 형식을 파싱"""
        # 1. 직접 JSON 파싱 시도
        try:
            parsed_json = json.loads(response_str)
            return (
                parsed_json.get('refined_query'),
                parsed_json.get('action'),
                parsed_json.get('classification'),
                parsed_json.get('is_new_topic'),
                parsed_json.get('reasoning')
            )
        except json.JSONDecodeError:
            pass
        
        # 2. 마크다운에서 JSON 추출 시도
        json_content = self._extract_json_from_markdown(response_str)
        if json_content:
            try:
                parsed_json = json.loads(json_content)
                return (
                    parsed_json.get('refined_query'),
                    parsed_json.get('action'),
                    parsed_json.get('classification'),
                    parsed_json.get('is_new_topic'),
                    parsed_json.get('reasoning')
                )
            except json.JSONDecodeError:
                pass
        
        # 3. 텍스트에서 직접 추출 시도 (fallback)
        import re
        
        # refined_query 추출
        query_match = re.search(r'refined_query["\s:]*([^,\n}]+)', response_str, re.IGNORECASE)
        extracted_query = query_match.group(1).strip().strip('"\'') if query_match else None
        
        # action 추출
        action_match = re.search(r'action["\s:]*([^,\n}]+)', response_str, re.IGNORECASE)
        extracted_action = action_match.group(1).strip().strip('"\'') if action_match else None
        
        # classification 추출
        class_match = re.search(r'classification["\s:]*([^,\n}]+)', response_str, re.IGNORECASE)
        extracted_class = class_match.group(1).strip().strip('"\'') if class_match else None
        
        # is_new_topic 추출
        topic_match = re.search(r'is_new_topic["\s:]*([^,\n}]+)', response_str, re.IGNORECASE)
        extracted_topic = None
        if topic_match:
            topic_str = topic_match.group(1).strip().strip('"\'').lower()
            extracted_topic = topic_str == 'true'
        
        # reasoning 추출
        reason_match = re.search(r'reasoning["\s:]*([^}]+)', response_str, re.IGNORECASE)
        extracted_reason = reason_match.group(1).strip().strip('"\'') if reason_match else None
        
        return extracted_query, extracted_action, extracted_class, extracted_topic, extracted_reason

    def _create_personalized_context(self, docs_text: str, user_data: Dict[str, Any] = None) -> str:
        """사용자 정보 기반 개인화된 컨텍스트 생성"""
        personalized_context = docs_text
#         if user_data and isinstance(user_data, dict):
#             user_context = f"""
# ###사용자 정보:
# - 이름: {user_data.get('userName', 'N/A')}
# - 연소득: {user_data.get('income', 'N/A')}원

# ###검색문서:
# """
#             # 보유 카드 정보 추가
#             user_data_obj = user_data.get('data', {})
#             if user_data_obj and isinstance(user_data_obj, dict) and user_data_obj.get('ownCardArr'):
#                 user_context += "\n보유 카드:\n"
#                 for card in user_data_obj['ownCardArr']:
#                     if isinstance(card, dict):
#                         user_context += f"- {card.get('bank', 'N/A')} {card.get('name', 'N/A')} ({card.get('type', 'N/A')}, 결제일: {card.get('paymentDate', 'N/A')}일)\n"
#             else:
#                 print(f"� [DEBUG] 카드 정보 없음 또는 조건 불만족")

#             print(f"�🔍 개인화된 컨텍스트: {user_context}")
#             personalized_context = user_context + "\n\n관련 문서:\n" + docs_text
#             print(f"🔍 개인화된 컨텍스트 길이: {len(personalized_context)} 문자")
#         else:
#             print(f"🐛 [DEBUG] user_data 조건 실패! user_data가 None이거나 dict가 아님")
        
        return personalized_context

    def _log_to_langfuse(self, trace, operation: str, input_data: str, output_data: str, metadata: Dict[str, Any]):
        """Langfuse 로깅 헬퍼"""
        if trace and self.langfuse_manager:
            self.langfuse_manager.log_generation(
                trace_context=trace.get('trace_context'),
                name=operation,
                input=input_data,
                output=output_data,
                metadata=metadata
            )

    # =============================================================================
    # 메인 질의 처리 메서드
    # =============================================================================



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
                
                # user_data가 없거나 유효하지 않은 경우에만 테스트 데이터 사용
                current_user_data = user_data
                if not current_user_data or not isinstance(current_user_data, dict):
                    print("🔧 [DEBUG] user_data가 없어서 기본 테스트 데이터 사용")
                    current_user_data = {  
                        "userId": "bccard",  
                        "userName": "김명정",  
                        "loginTime": "2025-08-27T14:23:45.123Z",
                        "isAuthenticated": True,
                        "age": "27",
                        "income": "77,511,577",
                        "data": {
                            "email": "kmj@deotis.co.kr",
                            "phone": "010-1234-5678",
                            "ownCardArr": [
                                {
                                    "bank": "우리카드",
                                    "paymentDate": "4",
                                    "type": "신용카드"
                                }
                            ]
                        }
                    }
                else:
                    print("🔧 [DEBUG] 전달받은 user_data 사용")

                trace = self._create_langfuse_trace(session_id, current_user_data)
                
                # 질의 분석 및 재정의
                analysis_result = self._analyze_and_refine_query(query, session_id, current_user_data)

                # DIRECT_ANSWER 처리
                if analysis_result['action'] == "DIRECT_ANSWER":
                    return self._handle_direct_answer(query, analysis_result, session_id, trace, start_time)
                
                # 검색 및 답변 생성
                return self._handle_search_answer(query, analysis_result, session_id, current_user_data, trace, start_time)
                
            except Exception as e:
                return self._handle_error(e, trace if 'trace' in locals() else None, start_time)

        return await asyncio.get_event_loop().run_in_executor(None, _process_query)
    
    def _create_langfuse_trace(self, session_id: str, user_data: Dict[str, Any] = None):
        """Langfuse 트레이스 생성"""
        trace = None
        if self.langfuse_manager:
            trace_metadata = {
                "session_id": session_id,
                "model": self.model_choice,
                "top_k": self.top_k
            }
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
        return trace

    def _analyze_and_refine_query(self, query: str, session_id: str, user_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """질의 분석 및 재정의"""
        chat_manager = self.get_chat_manager(session_id)
        history = chat_manager.build_history()
        
        print(f"🔍 대화 기록: {history}")
        print(f"🔍 원본 질의: {query}")
        
        userinfo_str = json.dumps(user_data, ensure_ascii=False, indent=2)
        
        try:
            refined_query_str = self.refinement_chain.run({
                "question": query, 
                "context": history, 
                "userinfo": userinfo_str
            })
        except Exception as chain_error:
            print(f"❌ refinement_chain 실행 오류: {str(chain_error)}")
            refined_query_str = query
            
        print(f"🔍 질의 분석 결과 (원본): {refined_query_str}")
        
        # 기본값 설정
        result = {
            "refined_query": query,
            "action": "SEARCH",
            "classification": "GENERAL",
            "is_new_topic": True,
            "reasoning": ""
        }
        
        try:
            parsed_result = self._parse_query_analysis(refined_query_str)
            parsed_refined_query, parsed_action, parsed_classification, parsed_is_new_topic, parsed_reasoning = parsed_result
            
            # 파싱된 값들을 안전하게 적용
            if parsed_refined_query:
                result["refined_query"] = parsed_refined_query
                print(f"🔍 정제된 질의: {parsed_refined_query}")
            
            if parsed_action in ["SEARCH", "DIRECT_ANSWER"]:
                result["action"] = parsed_action
                print(f"📋 처리 방식: {parsed_action}")
            
            if parsed_classification in ["GENERAL", "HYBRID", "USER_INFO_ONLY"]:
                result["classification"] = parsed_classification
                print(f"🏷️ 질의 분류: {parsed_classification}")
            
            if parsed_is_new_topic is not None:
                result["is_new_topic"] = parsed_is_new_topic
                print(f"🆕 새로운 주제: {parsed_is_new_topic}")
            
            if parsed_reasoning:
                result["reasoning"] = parsed_reasoning
                print(f"💭 분석 근거: {parsed_reasoning}")
            
            # 새로운 주제인 경우 대화 기록 초기화
            if result["is_new_topic"]:
                chat_manager.clear_history()
                print("🔄 대화 기록 초기화됨 (새로운 주제)")
                
        except Exception as e:
            print(f"❌ 질의 분석 파싱 실패: {str(e)}")
            print("🔄 기본값으로 처리 진행")
        
        return result
    
    def _handle_direct_answer(self, query: str, analysis_result: Dict[str, Any], session_id: str, trace, start_time: float) -> Dict[str, Any]:
        """DIRECT_ANSWER 액션 처리"""
        processing_time = time.time() - start_time
        direct_answer = analysis_result["refined_query"]
        
        print(f"🔍 직접 답변 모드 (프롬프트 생성): {direct_answer}")
        
        # 🎯 관련 링크 자동 추가
        enhanced_answer = add_related_links(direct_answer, query)
    
        # 대화 기록에 질문과 답변 추가
        chat_manager = self.get_chat_manager(session_id)
        chat_manager.add_chat(query, enhanced_answer)
        
        # Langfuse 로깅
        self._log_to_langfuse(trace, "direct_answer_generation", query, direct_answer, {
            "processing_time": processing_time,
            "mode": "direct_answer",
            "classification": analysis_result["classification"],
            "model": self.model_choice
        })
        
        return {
            "status": "success",
            "answer": enhanced_answer,
            "query": query,
            "refined_query": analysis_result["refined_query"],
            "classification": analysis_result["classification"],
            "action": analysis_result["action"],
            "reasoning": analysis_result["reasoning"],
            "session_id": session_id,
            "processing_time": processing_time,
            "retrieved_docs": []
        }
    
    def _handle_search_answer(self, query: str, analysis_result: Dict[str, Any], session_id: str, user_data: Dict[str, Any], trace, start_time: float) -> Dict[str, Any]:
        """검색 기반 답변 처리"""
        # 고도화된 하이브리드 검색
        merged_docs = self.retriever.get_relevant_documents(analysis_result["refined_query"])
        print(f"🔍 고도화된 하이브리드 검색 결과 개수: {len(merged_docs)}")

        # 문서를 텍스트로 변환
        docs_text = "\n\n---\n\n".join([
            getattr(doc, "page_content", str(doc)) for doc in merged_docs
        ])
        # 문서 텍스트 변환 다른 시도
        # docs_text = self.convert_docs_with_table_preservation(merged_docs)

        print(f"🔍 최종 문서 컨텍스트 길이: {len(docs_text)} 문자")

        # 개인화된 컨텍스트 생성
        personalized_context = self._create_personalized_context(docs_text, user_data)

        # 검색된 자료와 재정의 질문을 LLM에 넘겨서 답변 생성
        result = self.qa_chain.invoke({
            "question": analysis_result["refined_query"], 
            "context": personalized_context
        })

        print(f"🔍 RAG 체인 응답 구조: {result}")
        processing_time = time.time() - start_time

        # 답변 추출
        if result and ('answer' in result or 'result' in result or 'text' in result):
            original_answer = result.get('answer') or result.get('result') or result.get('text')
            print(f"🔍 원본 답변: {original_answer}")
            
            # 🎯 관련 링크 자동 추가
            enhanced_answer = add_related_links(original_answer, query)
            print(f"🔗 링크 추가된 최종 답변: {enhanced_answer}")
            
            # 대화 기록에 질문과 답변 추가
            chat_manager = self.get_chat_manager(session_id)
            chat_manager.add_chat(query, enhanced_answer)

            # Langfuse 로깅
            self._log_to_langfuse(trace, "rag_generation", query, enhanced_answer, {
                "processing_time": processing_time,
                "retrieved_docs_count": len(merged_docs),
                "model": self.model_choice
            })

            # retrieved_docs 구성
            retrieved_docs = []
            for doc in merged_docs:
                retrieved_docs.append({
                    "type": "enhanced_hybrid",
                    "content": getattr(doc, "page_content", str(doc)),
                    "metadata": getattr(doc, "metadata", {})
                })

            return {
                "status": "success",
                "answer": enhanced_answer,
                "query": query,
                "refined_query": analysis_result["refined_query"],
                "classification": analysis_result["classification"],
                "action": analysis_result["action"],
                "reasoning": analysis_result["reasoning"],
                "session_id": session_id,
                "username": user_data.get("userName", "") if user_data and isinstance(user_data, dict) else "",
                "processing_time": processing_time,
                "retrieved_docs": retrieved_docs
            }
        else:
            return self._handle_no_answer_error(result, trace, processing_time)
    
    def _handle_no_answer_error(self, result, trace, processing_time: float) -> Dict[str, Any]:
        """답변 생성 실패 처리"""
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
    
    def _handle_error(self, error: Exception, trace, start_time: float) -> Dict[str, Any]:
        """에러 처리"""
        processing_time = time.time() - start_time
        
        if trace and self.langfuse_manager:
            self.langfuse_manager.log_event(
                trace_context=trace.get('trace_context'),
                name="rag_exception",
                metadata={
                    "error": str(error),
                    "processing_time": processing_time
                }
            )

        return {
            "status": "error",
            "message": f"질의 처리 오류: {str(error)}",
            "processing_time": processing_time
        }

# =============================================================================
# 애플리케이션 라이프사이클 관리
# =============================================================================
    
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

    # =============================================================================
    # Elasticsearch 인덱스 관리 메서드들
    # =============================================================================
    
    def change_elasticsearch_index(self, new_index_name: str) -> bool:
        """Elasticsearch 인덱스를 동적으로 변경하는 메서드 (완전한 버전)"""
        try:
            # 새 인덱스 이름 유효성 검사
            if not new_index_name or not isinstance(new_index_name, str):
                raise ValueError("인덱스 이름이 유효하지 않습니다.")
            
            # 특수문자 검사 (Elasticsearch 인덱스 이름 규칙)
            import re
            if not re.match(r'^[a-z0-9_-]+$', new_index_name.lower()):
                raise ValueError("인덱스 이름은 소문자, 숫자, '_', '-'만 사용할 수 있습니다.")
            
            old_index = self.current_index_name
            print(f"🔄 인덱스 변경 시작: '{old_index}' → '{new_index_name}'")
            
            # 🎯 핵심: Retriever 완전 재생성
            try:
                # 기존 retriever 백업 (실패 시 복원용)
                old_retriever = self.retriever if hasattr(self, 'retriever') else None
                
                # 시스템이 초기화되어 있는지 확인
                if not self.is_initialized or not self.embedding_model:
                    print("❌ 시스템이 초기화되지 않았습니다. 먼저 initialize를 수행하세요.")
                    return False
                
                # 새 인덱스로 Enhanced Retriever 재생성
                from core.rag import create_enhanced_retriever
                
                print(f"🔍 새 Retriever 생성 중... (인덱스: {new_index_name})")
                
                # index_name 파라미터를 명시적으로 전달
                new_retriever = create_enhanced_retriever(
                    embedding_model=self.embedding_model,
                    top_k=self.top_k,
                    index_name=new_index_name  # 🎯 새 인덱스 명시적 전달
                )
                
                if new_retriever is None:
                    raise Exception(f"인덱스 '{new_index_name}'로 Retriever 생성 실패")
                
                # 성공하면 교체
                self.retriever = new_retriever
                self.current_index_name = new_index_name
                
                print(f"✅ 인덱스 변경 완료: '{new_index_name}'")
                print(f"✅ Retriever 재생성 완료")
                
                return True
                
            except Exception as retriever_error:
                # 실패 시 기존 retriever 복원
                print(f"❌ Retriever 재생성 실패: {str(retriever_error)}")
                print(f"🔄 기존 인덱스 '{old_index}'로 복원")
                
                if old_retriever is not None:
                    self.retriever = old_retriever
                
                return False
            
        except Exception as e:
            print(f"❌ 인덱스 변경 중 오류 발생: {str(e)}")
            return False
    
    def get_current_index(self) -> str:
        """현재 사용 중인 Elasticsearch 인덱스 이름을 반환"""
        return self.current_index_name
    
    def list_available_indices(self) -> list:
        """사용 가능한 Elasticsearch 인덱스 목록을 반환"""
        try:
            if hasattr(self, 'retriever') and self.retriever and hasattr(self.retriever, 'client'):
                es_client = self.retriever.client
                if es_client and hasattr(es_client, 'indices'):
                    # 모든 인덱스 조회
                    indices = es_client.indices.get_alias(index="*")
                    return list(indices.keys())
            return ["test2_rag", "yang_deotis_rag"]  # 기본값
        except Exception as e:
            print(f"인덱스 목록 조회 중 오류: {str(e)}")
            return ["test2_rag", "yang_deotis_rag"]  # 기본값


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


# =============================================================================
# Langfuse 설정 함수들
# =============================================================================


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
