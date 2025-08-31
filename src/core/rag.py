"""
RAG 시스템 핵심 로직
"""
import os
from typing import Tuple, Union, List
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from core.config import ELASTICSEARCH_URL, INDEX_NAME
from utils.elasticsearch import ElasticsearchManager


def check_ollama_connection() -> Tuple[bool, str]:
    """Ollama 서버 연결 상태 확인"""
    try:
        import requests
        import socket
        from urllib.parse import urlparse
        
        # 환경 변수에서 Ollama URL 가져오기
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        parsed_url = urlparse(ollama_url)
        host = parsed_url.hostname or 'localhost'
        port = parsed_url.port or 11434
        
        # 1. HTTP 응답 확인 (더 신뢰할 수 있는 방법)
        try:
            response = requests.get(f"{ollama_url}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                return True, f"Ollama 서버 연결 성공 (버전: {version_info.get('version', 'unknown')})"
            else:
                return False, f"Ollama 서버 응답 오류 (상태 코드: {response.status_code})"
        except requests.exceptions.ConnectionError:
            # HTTP 요청이 실패하면 소켓으로 포트 확인
            pass
        except requests.exceptions.Timeout:
            return False, "Ollama 서버 응답 타임아웃"
        
        # 2. 소켓 연결 확인 (HTTP 실패 시에만)
        # 여러 소켓 패밀리 시도 (IPv4, IPv6)
        for family in [socket.AF_INET, socket.AF_INET6]:
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    return False, f"Ollama 서버 포트는 열려있지만 HTTP 응답이 없습니다"
            except Exception:
                continue
        
        return False, f"Ollama 서버가 실행되지 않고 있습니다 (포트 {port} 닫힘)"
            
    except ImportError:
        return False, "requests 라이브러리가 설치되지 않았습니다"
    except Exception as e:
        return False, f"Ollama 연결 확인 중 오류: {str(e)}"


def create_rag_chain(embeddings, llm_model, top_k: int = 3, callbacks=None) -> Tuple[Union[RetrievalQA, None], Union[bool, str]]:
    """RAG 체인 생성"""
    try:
        print("🚀 RAG 체인 생성 시작...")
        
        # 0. Ollama 서버 연결 사전 확인 (LLM 모델이 Ollama 기반인 경우)
        try:
            # LLM 모델이 Ollama 기반인지 확인
            if hasattr(llm_model, '_client') or 'ollama' in str(type(llm_model)).lower():
                print("🔍 Ollama 기반 LLM 모델 감지, 연결 상태 확인 중...")
                ollama_connected, ollama_message = check_ollama_connection()
                if not ollama_connected:
                    return None, f"Ollama 서버 연결 실패: {ollama_message}\n\n🔧 해결 방법:\n1. Ollama 다운로드: https://ollama.ai/download\n2. Ollama 시작: ollama serve\n3. 모델 설치: ollama pull qwen2:7b\n4. 확인: ollama list"
                print(f"✅ {ollama_message}")
        except Exception as ollama_check_error:
            print(f"⚠️ Ollama 연결 확인 실패 (계속 진행): {str(ollama_check_error)}")
        
        # 1. Elasticsearch 클라이언트 연결 확인 - 상세한 예외 처리
        try:
            print("🔍 Elasticsearch 연결 확인 중...")
            es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
            if not success:
                return None, f"Elasticsearch 연결 실패: {message}\n해결 방법:\n1. Elasticsearch 서버 시작: docker-compose up -d elasticsearch\n2. 포트 확인: {ELASTICSEARCH_URL}\n3. 방화벽 설정 확인"
            print(f"✅ Elasticsearch 연결 성공: {message}")
        except ConnectionError as es_conn_error:
            return None, f"Elasticsearch 연결 오류: {str(es_conn_error)}\nElasticsearch 서버가 실행 중인지 확인하세요: {ELASTICSEARCH_URL}"
        except Exception as es_error:
            return None, f"Elasticsearch 클라이언트 생성 오류: {str(es_error)}"
        
        # 인덱스 존재 확인 - 상세한 예외 처리
        try:
            print(f"📋 인덱스 '{INDEX_NAME}' 존재 확인 중...")
            if not es_client.indices.exists(index=INDEX_NAME):
                return None, f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다.\n해결 방법:\n1. PDF 파일을 pdf/ 디렉토리에 추가\n2. PDF 인덱싱 실행\n3. 또는 기존 인덱스 이름 확인"
            print(f"✅ 인덱스 '{INDEX_NAME}' 존재 확인")
        except Exception as idx_error:
            return None, f"인덱스 확인 중 오류: {str(idx_error)}"
        
        # 문서 개수 확인 - 상세한 예외 처리
        try:
            print("📊 인덱스 문서 개수 확인 중...")
            doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
            if doc_count == 0:
                return None, f"인덱스 '{INDEX_NAME}'에 문서가 없습니다 (문서 수: {doc_count})\n해결 방법:\n1. PDF 파일 인덱싱 실행\n2. 인덱스 데이터 확인"
            print(f"✅ 인덱스에 {doc_count}개 문서 존재")
        except Exception as count_error:
            return None, f"문서 개수 확인 중 오류: {str(count_error)}"
        
        # Elasticsearch 벡터스토어 연결 - 상세한 예외 처리
        try:
            print("🔗 Elasticsearch 벡터스토어 연결 중...")
            # ElasticsearchStore에 인증이 포함된 URL 사용
            vectorstore = ElasticsearchStore(
                embedding=embeddings,
                index_name=INDEX_NAME,
                es_url=ELASTICSEARCH_URL
            )
            print("✅ 벡터스토어 연결 성공 (es_url 방식)")
        except TypeError as type_error:
            print("⚠️ es_url 파라미터 오류, elasticsearch_url로 재시도...")
            # 파라미터 이름 문제인 경우
            try:
                vectorstore = ElasticsearchStore(
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                    elasticsearch_url=ELASTICSEARCH_URL
                )
                print("✅ 벡터스토어 연결 성공 (elasticsearch_url 방식)")
            except ConnectionError as conn_error2:
                return None, f"벡터스토어 생성 실패 - 연결 오류 (elasticsearch_url): {str(conn_error2)}"
            except TimeoutError as timeout_error2:
                return None, f"벡터스토어 생성 실패 - 타임아웃 (elasticsearch_url): {str(timeout_error2)}"
            except Exception as vs_error2:
                import traceback
                error_traceback = traceback.format_exc()
                return None, f"벡터스토어 생성 실패 (elasticsearch_url):\n예외 타입: {type(vs_error2).__name__}\n오류 메시지: {str(vs_error2)}\n스택 트레이스:\n{error_traceback}"
        except ConnectionError as conn_error:
            return None, f"벡터스토어 생성 실패 - Elasticsearch 연결 오류: {str(conn_error)}\nElasticsearch 서버가 실행 중인지 확인하세요: {ELASTICSEARCH_URL}"
        except TimeoutError as timeout_error:
            return None, f"벡터스토어 생성 실패 - 연결 타임아웃: {str(timeout_error)}\nElasticsearch 서버 응답이 느립니다: {ELASTICSEARCH_URL}"
        except ImportError as import_error:
            return None, f"벡터스토어 생성 실패 - 라이브러리 import 오류: {str(import_error)}\n필요한 패키지가 설치되지 않았을 수 있습니다."
        except Exception as vs_error:
            import traceback
            error_traceback = traceback.format_exc()
            return None, f"벡터스토어 생성 실패 - 예상치 못한 오류:\n예외 타입: {type(vs_error).__name__}\n오류 메시지: {str(vs_error)}\nElasticsearch URL: {ELASTICSEARCH_URL}\n스택 트레이스:\n{error_traceback}"
        
        # 리트리버 설정 - 하이브리드 검색으로 품질 향상
        try:
            print("🔄 리트리버 설정 중...")
            
            # 기본 시맨틱 검색 + 키워드 검색 조합
            base_retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k * 2,  # 더 많은 후보 문서 확보
                    "fetch_k": min(top_k * 6, 10000)  # 초기 검색 범위 확대
                }
            )
            
            # 하이브리드 검색을 위한 키워드 검색 연결
            
            # 리랭킹을 위한 컨텍스트 기반 필터링 추가 (할루시네이션 방지 강화)
            def enhanced_retrieve(query):
                # 1차: 기본 시맨틱 검색
                semantic_docs = base_retriever.get_relevant_documents(query)
                
                # 2차: 키워드 검색으로 보완
                keyword_results = ElasticsearchManager.keyword_search(query, top_k * 2)
                
                # 3차: 관련성 검증 및 키워드 매칭 강화
                query_keywords = query.lower().split()
                scored_docs = []
                
                for doc in semantic_docs:
                    content = doc.page_content.lower()
                    
                    # 기본 키워드 매칭 점수
                    keyword_score = sum(1 for keyword in query_keywords if keyword in content)
                    
                    # 관련성 임계값 설정 (할루시네이션 방지)
                    min_relevance_score = 0.1  # 최소 관련성 점수
                    if keyword_score == 0 and len(doc.page_content) < 50:
                        # 키워드 매칭이 전혀 없고 내용이 너무 짧으면 제외
                        continue
                    
                    metadata_score = 0
                    
                    # 메타데이터 기반 스코어링
                    if hasattr(doc, 'metadata'):
                        if doc.metadata.get('structure_type') == '업무안내서' and '업무' in query:
                            metadata_score += 2
                        if doc.metadata.get('has_tables') and ('표' in query or '목록' in query):
                            metadata_score += 1
                        if doc.metadata.get('category') == 'DOCX' and ('안내' in query or '절차' in query):
                            metadata_score += 1.5
                    
                    # 키워드 검색 결과와 매칭되면 보너스 점수
                    filename = doc.metadata.get('filename', '')
                    for kw_result in keyword_results:
                        if kw_result.get('metadata', {}).get('filename') == filename:
                            metadata_score += 1
                            break
                    
                    total_score = keyword_score + metadata_score
                    
                    # 최소 점수 이상인 문서만 포함 (할루시네이션 방지)
                    if total_score >= min_relevance_score:
                        scored_docs.append((doc, total_score))
                
                # 스코어 기반 정렬 후 상위 k개 반환
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                final_docs = [doc for doc, _ in scored_docs[:top_k]]
                
                # 빈 결과 처리 (할루시네이션 방지)
                if not final_docs:
                    print("⚠️ 관련성 있는 문서를 찾을 수 없습니다.")
                
                return final_docs
            
            # 커스텀 리트리버 클래스 생성
            class EnhancedRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str, *, run_manager=None):
                    return enhanced_retrieve(query)
                
                def get_relevant_documents(self, query):
                    return enhanced_retrieve(query)
            
            retriever = EnhancedRetriever()
            print(f"✅ 향상된 하이브리드 리트리버 설정 완료 (시맨틱 + 키워드 검색, top_k: {top_k})")
        except AttributeError as attr_error:
            return None, f"리트리버 설정 실패 - 속성 오류: {str(attr_error)}\n벡터스토어 객체가 올바르지 않을 수 있습니다."
        except ValueError as value_error:
            return None, f"리트리버 설정 실패 - 값 오류: {str(value_error)}\nsearch_kwargs 설정을 확인하세요."
        except Exception as ret_error:
            return None, f"리트리버 설정 실패 - 예상치 못한 오류: {str(ret_error)}"

        prompt_template = """
    당신은 제공된 문서만을 기반으로 정확하고 친절하게 답변하는 AI 어시스턴트입니다.
    
    **[중요 규칙]**
    1. 반드시 아래 제공된 문서 내용만을 사용하여 답변해야 합니다.
    2. 문서에 없는 내용이나 사전 지식은 절대 추가하지 마세요.
    3. 질문에 관련된 문서 내용이 없으면, "죄송하지만, 제공된 문서 내에서는 질문과 관련된 정보를 찾을 수 없습니다."라고 명확하게 답변하세요.
    4. 질문이 모호하거나 여러 해석이 가능한 경우, 반드시 구체적 재질의를 요청하세요.
    
    **[답변 방식]**
    - 사용자의 질문 의도를 파악해, 문서에서 최대한 정확하고 친절한 답변을 제공합니다.
    - 질문이 포괄적이긴 하지만 일반 답변이 가능한 경우, 먼저 답변한 뒤 문서 기반 추가 궁금증이나 후속 질문을 2~3개 제안합니다.
    - 만약 질문이 너무 모호해 일반 답변조차 어려우면, '재질의 요청 규칙'에 따라 옵션을 제안해 질문을 구체화할 수 있도록 유도합니다.
    
    **[재질의 요청 규칙 및 형식]**
    - 질문이 너무 일반적이거나 여러 서비스에 걸쳐 있을 때, 혹은 핵심 키워드가 부족할 때 사용합니다.
    - 형식: 
    "더 정확한 답변을 위해 다음 중 어떤 점에 대해 구체적으로 알고 싶으신지 선택해 주세요:
    1. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 1]
    2. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 2]
    3. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 3]"
    
    **[제공된 문서 내용]**
    {context}
    
    **[질문]**
    {question}
    
    **[답변]**

"""
        
        try:
            print("📝 프롬프트 템플릿 설정 중...")
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            print("✅ 프롬프트 템플릿 설정 완료")
        except ValueError as prompt_value_error:
            return None, f"프롬프트 템플릿 설정 실패 - 값 오류: {str(prompt_value_error)}\ninput_variables 설정을 확인하세요."
        except Exception as prompt_error:
            return None, f"프롬프트 템플릿 설정 실패: {str(prompt_error)}"
        
        # QA 체인 생성 - 상세한 예외 처리
        try:
            print("⚙️ QA 체인 생성 중...")
            
            # callbacks가 제공된 경우 포함
            qa_kwargs = {
                "llm": llm_model,
                "chain_type": "stuff",
                "retriever": retriever,
                "chain_type_kwargs": {"prompt": prompt},
                "return_source_documents": True
            }
            
            if callbacks:
                qa_kwargs["callbacks"] = callbacks
                print(f"✅ Langfuse 콜백 추가됨")
            
            qa_chain = RetrievalQA.from_chain_type(**qa_kwargs)
            print("✅ QA 체인 생성 완료")
        except AttributeError as qa_attr_error:
            return None, f"QA 체인 생성 실패 - 속성 오류: {str(qa_attr_error)}\nLLM 모델이나 리트리버가 올바르지 않을 수 있습니다."
        except ValueError as qa_value_error:
            return None, f"QA 체인 생성 실패 - 값 오류: {str(qa_value_error)}\nchain_type이나 매개변수를 확인하세요."
        except ImportError as qa_import_error:
            return None, f"QA 체인 생성 실패 - import 오류: {str(qa_import_error)}\nLangChain 라이브러리를 확인하세요."
        except Exception as qa_error:
            return None, f"QA 체인 생성 실패 - 예상치 못한 오류: {str(qa_error)}"
        
        # 최종 검증
        if qa_chain is None:
            return None, "QA 체인이 None으로 생성되었습니다. 생성 과정에서 문제가 발생했습니다."
        
        # 간단한 테스트 쿼리 - 상세한 예외 처리
        try:
            print("🔍 QA 체인 테스트 중...")
            test_result = qa_chain({"query": "카드 이용한도 조정사유"})
            if test_result is None:
                return None, "QA 체인 테스트 실패: 응답이 None입니다."
            print("✅ QA 체인 테스트 성공")
        except ConnectionError as conn_error:
            if "10061" in str(conn_error):
                return None, f"QA 체인 테스트 실패 - Ollama 서버 연결 오류: {str(conn_error)}\n\n해결 방법:\n1. Ollama 설치: https://ollama.ai/download\n2. Ollama 서비스 시작: ollama serve\n3. 모델 설치: ollama pull qwen2:7b\n4. 서비스 확인: ollama list"
            else:
                return None, f"QA 체인 테스트 실패 - 연결 오류: {str(conn_error)}"
        except TimeoutError as timeout_error:
            return None, f"QA 체인 테스트 실패 - 타임아웃: {str(timeout_error)}\nOllama 서버 응답이 느립니다. 서버 상태를 확인하세요."
        except ImportError as import_error:
            return None, f"QA 체인 테스트 실패 - 모듈 import 오류: {str(import_error)}\n필요한 패키지: pip install ollama langchain-ollama"
        except AttributeError as attr_error:
            if "ollama" in str(attr_error).lower():
                return None, f"QA 체인 테스트 실패 - Ollama 속성 오류: {str(attr_error)}\nOllama 클라이언트 설정을 확인하세요."
            else:
                return None, f"QA 체인 테스트 실패 - 속성 오류: {str(attr_error)}"
        except ValueError as value_error:
            return None, f"QA 체인 테스트 실패 - 값 오류: {str(value_error)}\n모델 설정이나 매개변수를 확인하세요."
        except KeyError as key_error:
            return None, f"QA 체인 테스트 실패 - 키 오류: {str(key_error)}\n설정 파일이나 환경 변수를 확인하세요."
        except Exception as test_error:
            import traceback
            error_traceback = traceback.format_exc()
            
            # httpx.ConnectError 특별 처리
            if "httpx.ConnectError" in error_traceback or "WinError 10061" in str(test_error):
                return None, f"QA 체인 테스트 실패 - Ollama 연결 오류:\n오류: {str(test_error)}\n\n🔧 해결 방법:\n1. Ollama 다운로드 및 설치: https://ollama.ai/download\n2. Ollama 서비스 시작:\n   - Windows: ollama serve (또는 자동 시작)\n   - 또는 Windows 서비스에서 Ollama 시작\n3. 모델 다운로드:\n   - ollama pull qwen2:7b\n   - ollama pull solar:10.7b\n4. 설치 확인: ollama list\n5. 포트 확인: netstat -an | findstr 11434"
            else:
                return None, f"QA 체인 테스트 실패 - 예상치 못한 오류:\n예외 타입: {type(test_error).__name__}\n오류 메시지: {str(test_error)}\n상세 스택 트레이스:\n{error_traceback}"
        
        return qa_chain, True
        
    except ConnectionError as conn_error:
        return None, f"연결 오류: {str(conn_error)}\nElasticsearch 서버가 실행 중인지 확인하세요."
    except TimeoutError as timeout_error:
        return None, f"타임아웃 오류: {str(timeout_error)}\nElasticsearch 서버 응답이 느립니다."
    except ImportError as import_error:
        return None, f"라이브러리 import 오류: {str(import_error)}\n필요한 패키지를 설치하세요: pip install -r requirements.txt"
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return None, f"예상치 못한 오류:\n예외 타입: {type(e).__name__}\n오류 메시지: {str(e)}\n상세 스택 트레이스:\n{error_traceback}"

# 전역 프롬프트 템플릿 변수 (내용은 직접 작성)
prompt_for_refined_query = """
당신은 사용자의 질문 의도를 파악하고, 후속 검색 단계에 가장 효과적인 형태로 질문을 재구성하는 전문 AI 어시스턴트입니다.
주어진 **[사용자 정보]**, **[대화 기록]**, **[최신 사용자 질문]**을 바탕으로, 아래 정의된 단계와 규칙에 따라 분석하고 JSON 형식으로 결과를 출력해 주세요.

### 역할 및 핵심 목표
1.  **맥락 분석**: 현재 질문이 이전 대화의 연장선인지, 새로운 주제인지 판단합니다.
2.  **정보 소스 분류**: 질문에 답하기 위해 어떤 정보(사용자 정보, 검색 문서)가 필요한지 분류합니다.
3.  **다음 행동 결정**: 분석 결과를 바탕으로 시스템이 취해야 할 다음 행동(`action`)을 결정하고, 필요시 검색에 최적화된 `refined_query`를 생성합니다.

### 분석 및 처리 단계
**1단계: 대화 맥락 유지 여부 판단**
-   **[최신 사용자 질문]**이 **[대화 기록]**의 주제와 이어진다면 맥락이 '유지'된 것입니다.
-   관련 없는 새로운 주제라면 맥락이 '리셋'된 것입니다.

**2단계: 정보 소스 분류(classification)**

**중요한 분류 원칙**: 질문에서 언급된 카드사/카드 정보가 **[사용자 정보]**에 실제로 존재하는지 먼저 검증해야 합니다.

-   **GENERAL**: 사용자의 개인 정보와 무관하며, 오직 **[검색 문서]**를 통해서만 답변할 수 있는 일반적인 질문입니다.
        * 다음 조건 중 하나라도 만족하면 GENERAL로 분류합니다.
             - 질문에서 '내 카드', '내 소득', '내 카드 결제일', '제 카드', '제 소득', '제 카드 결제일' 등 명확한 소유 표현이 없는 경우
             - **구체적인 카드사명 + 결제일 언급 시**: 질문에서 구체적인 카드사명과 결제일이 함께 언급된 경우 **반드시 GENERAL로 분류**합니다.
                예: "우리카드 결제일 28일 이용기간" → GENERAL (구체적 조건 명시)
                예: "BC카드 결제일 3일 이용기간" → GENERAL (구체적 조건 명시)
                예: "SC제일은행 결제일자가 5일일때, 이용기간은?" → GENERAL (구체적 조건 명시)
             - **사용자 정보와 다른 카드사명 언급 시**: 사용자 정보에 포함되지 않은 카드사가 언급된 경우
                예: "삼성카드 혜택은?" (사용자가 우리카드만 보유한 경우) → GENERAL

-   **HYBRID**: 다음 **모든 조건**을 만족해야 HYBRID로 분류됩니다:
    * 조건1: 질문에 '내 카드', '내 소득', '내 카드 결제일', '제 카드', '제 소득', '제 카드 결제일' 등 명확한 소유표현이 있는 경우.
    * 조건2: 질문에서 언급된 카드사/카드명이 **[사용자 정보]**에 실제로 존재해야 함
    * 조건3: **[사용자 정보]**와 **[검색 문서]**를 모두 참조해야만 정확한 답변이 가능해야 함
    * 예: "내 카드 혜택은?" (사용자 카드 정보 + 문서 검색 필요) → HYBRID
    * 예: "우리카드 결제일이 4일일때 이용기간은?" (사용자가 우리카드 보유 + 결제일 4일인 경우) → HYBRID
    
-   **USER_INFO_ONLY**: **[검색 문서]** 없이, 오직 **[사용자 정보]**에 실제로 존재하는 정보만으로 충분히 답변할 수 있는 질문입니다.

**3단계: 다음 행동(`action`) 결정 및 `refined_query` 생성**
-   **만약 `USER_INFO_ONLY`로 분류된다면:**
    -   `action`은 `DIRECT_ANSWER`로 설정합니다.
    -   `refined_query`에는 **[사용자 정보]**를 바탕으로 생성한 **완전한 답변**을 제공합니다.
    -   예를 들어, 사용자가 "내 카드 결제일 언제야?"라고 물었을 때, 사용자 정보에서 결제일을 찾아 "고객님의 우리카드 결제일은 매월 4일입니다."와 같이 완전한 답변을 생성합니다.
-   **만약 `GENERAL` 또는 `HYBRID`로 분류된다면:**
    -   `action`은 `SEARCH`로 설정합니다.
    -   `refined_query`에는 검색엔진에 사용할 '검색 최적화 질의'를 생성합니다. 아래 규칙을 반드시 따릅니다.
        -   이전 대화 없이도 독립적으로 이해 가능하도록 완전한 문장 형태로 재구성합니다.
        
        **🎯 분류별 치환 규칙:**
        
        **A. GENERAL로 분류된 경우:**
        -   **원문 보존 원칙**: 사용자가 명시한 카드사명, 결제일, 조건 등을 **그대로 유지**하여 검색 쿼리를 생성합니다.
        -   **치환 금지**: 사용자 정보에 없는 카드사나 조건이 언급된 경우, 절대 사용자 정보로 치환하지 마세요.
        -   예시:
            * "BC카드 결제일 3일 이용기간" → "BC카드 결제일 3일 이용기간" (원문 그대로)
            * "SC제일은행 결제일 5일 이용기간" → "SC제일은행 결제일 5일 이용기간" (원문 그대로)
            * "삼성카드 혜택" → "삼성카드 혜택" (원문 그대로)
        
        **B. HYBRID로 분류된 경우:**
        -   **카드 정보 치환 규칙 - 키워드 기반 확장**: '내 카드', '나의 카드', '보유 카드', '제 카드' 등이 포함된 경우, 아래 키워드 매칭 규칙에 따라 **[사용자 정보]**의 카드사명과 카드타입으로 치환합니다.
            
            **키워드 매칭 규칙:**
            - **우리카드만 치환**: 질문에 다음 키워드 중 하나가 포함된 경우
                * 키워드: 결제일, 결제일자, 이용기간, 결제 예정일
                * 예: "내 카드 결제일자는?" → "우리카드 결제일자는?"
            
            - **신용카드만 치환**: 질문에 다음 키워드 중 하나가 포함된 경우
                * 키워드: 가족카드, 한도, 할부, 일시불, 서명, 혜택, 부가서비스, 포인트, 연회비, 대출, 이자율, 상환, 분실, 신고, 정지, 해지, 연체, 해외
                * 예: "내 카드 한도는?" → "신용카드 한도는?"
                * 예: "내 카드 혜택은?" → "신용카드 혜택은?"
            
            - **우리카드 + 신용카드 모두 치환**: 질문에 다음 키워드 중 하나가 포함된 경우
                * 키워드: 소득공제, 카드 발급, 선결제
                * 예: "내 카드 소득공제는?" → "우리카드 신용카드 소득공제는?"
                * 예: "내 카드 발급 조건은?" → "우리카드 신용카드 발급 조건은?"
            
            - **기본 치환**: 위 키워드에 해당하지 않는 일반적인 경우
                * 예: "내 카드 정보는?" → "신용카드 정보는?"
            
            **치환 실행 규칙:**
            * **실제 보유 카드 검증**: 먼저 질문에서 언급된 카드가 **[사용자 정보]**에 실제로 존재하는지 확인해야 합니다.
            * **최소한의 치환 원칙**: 질문에 명시적으로 언급되지 않은 정보는 키워드 매칭에 의해서만 추가합니다.
            * **원문 존중 원칙**: 사용자가 질문에서 언급한 용어와 표현을 최대한 보존하세요.
            * **다중 카드**: 여러 카드 보유 시에도 키워드 매칭 규칙에 따라 처리: "우리카드 OR 국민카드 혜택"
            
        -   **기타 개인정보 치환**: '내 나이', '내 소득' 등은 해당하는 실제 값으로 치환합니다.
        -   **이름 제외 원칙**: 사용자 '이름'은 답변에 필수적인 경우가 아니면 질의에 포함하지 않습니다.
        -   **맥락 활용**: '대화 이력'의 맥락을 적극 활용하여 질문을 재정의합니다.
        -   **부정어 처리 규칙 (매우 중요):**
                만약 사용자의 이전 요청을 재정의한 결과에 대해 '아니', '말고', '그거 말고'와 같은 부정어로 수정 지시를 한다면, 기존의 재정의 규칙을 취소하고 정반대의 규칙을 적용하여 질문을 다시 재정의하세요.
                    개인화된 질문에 대한 부정: 사용자가 재정의된 질문에 대해 "아니 내 카드 말고"라고 했다면, 이는 개인 맞춤 정보가 아닌 일반 정보를 원한다는 뜻입니다.
                    처리: 이전 단계에서 추가했던 [사용자 정보]를 제거하고, 일반적인 질문으로 재정의하세요.
                    예시:
                        원래 요청: "카드 결제일 알려줘."
                        재정의 (오류): "우리카드 결제일 알려줘"
                        사용자 수정 지시: "아니 내 카드 말고"
                        재수정: "카드 결제일 알려줘" (일반적인 정보)
                
### 출력 형식
반드시 아래 JSON 형식에 맞춰서 결과를 반환해야 합니다:
{{
  "is_new_topic": boolean, // 1단계 결과: 새로운 주제이면 true, 맥락이 이어지면 false
  "classification": "string", // 2단계 결과: GENERAL, HYBRID, USER_INFO_ONLY
  "action": "string", // 3단계 결과: SEARCH, DIRECT_ANSWER
  "refined_query": "string", // 3단계 결과: '즉답' 또는 '검색 최적화 질의'
  "reasoning": "string" // 위 모든 결정에 대한 종합적인 이유
}}

### 제약사항
- **고유명사 처리**: '우리카드', '국민카드', '제일은행', '하나카드' 등은 특정 금융사의 고유명사이므로, 특히 '우리(our) 카드'와 같은 일반 명사와 혼동하지 마세요. 기타 고유명사는 질문에 따라 추가될 수 있습니다.
- **출력 언어**: 모든 출력은 반드시 한글로 작성합니다.
- **간결성**: 응답에 불필요한 단어, 문구, 대화형 표현이나 수식어를 절대 포함하지 마세요.

---

### **[사용자 정보]** 데이터 구조 안내

**[사용자 정보]**는 아래 JSON 형태로 입력됩니다. 예시에서 card 정보가 여러 개 존재하는 경우 ownCardArr 배열에 여러 카드 정보가 포함될 수 있습니다.

{{
    "userId": "bccard",
    "userName": "김명정",
    "loginTime": "2025-08-27T14:23:45.123Z",
    "isAuthenticated": true,
    "age": "27",
    "income": "77,511,577",
    "data": {{
        "email": "kmj@deotis.co.kr",
        "phone": "010-1234-5678",
        "ownCardArr": [
        {{
            "bank": "우리카드",        // 은행/카드사명
            "paymentDate": "4",       // 결제일
            "type": "신용카드"        // 카드 타입
        }}
        ]
    }}
}}

---

### 예시

**# 예시 1: GENERAL (구체적인 카드사명 언급, 새로운 주제) - 원문 보존**
- **사용자 정보**: {{
    ...
    "data": {{
      ...
      "ownCardArr": [
        {{ "bank": "우리카드", "type": "신용카드", "paymentDate": "4" }}
      ]
    }}
  }}
- **대화 기록**: "사용자: 우리카드 혜택 알려줘 / 어시스턴트: 네, 우리카드는..."
- **최신 사용자 질문**: "BC카드 결제일자가 4일일때, 이용기간은?"
- **결과**:
{{
  "is_new_topic": true,
  "action": "SEARCH",
  "classification": "GENERAL",
  "refined_query": "BC카드 결제일 4일 이용기간",
  "reasoning": "사용자 정보에 BC카드가 없고 구체적인 카드사명(BC카드)과 결제일(3일)을 명시했으므로 GENERAL로 분류하고, 원문을 그대로 보존하여 검색 쿼리를 생성했습니다."
}}

**# 예시 2: HYBRID (맥락 유지) - 카드사명만 치환**
- **사용자 정보**: {{
    ...
    "data": {{
      ...
      "ownCardArr": [
        {{ "bank": "우리카드", "type": "신용카드", "paymentDate": "4" }}
      ]
    }}
  }}
- **대화 기록**: "사용자: 내 카드 혜택 알려줘 / 어시스턴트: 네, 카드는..."
- **최신 사용자 질문**: "내 카드의 결제일자와 이용기간을 알려줘"
- **결과**:
{{
  "is_new_topic": false,
  "action": "SEARCH",
  "classification": "HYBRID",
  "refined_query": "우리카드 결제일 4일 이용기간",
  "reasoning": "사용자 정보의 '우리카드', '4일'을 참조해야 하고, '이용기간' 정보는 문서 검색이 필요하므로 HYBRID로 분류했습니다. '내 카드'는 '우리카드'로만 치환하고, 원문에 없는 '신용카드' 타입은 추가하지 않았습니다."
}}

**# 예시 3: USER_INFO_ONLY (맥락 유지) - 카드사명만 사용**
- **사용자 정보**: {{
    ...
    "data": {{
      ...
      "ownCardArr": [
        {{ "bank": "우리카드", "type": "신용카드", "paymentDate": "4" }}
      ]
    }}
  }}
- **대화 기록**: "사용자: 내 카드 혜택 알려줘 / 어시스턴트: 네, 우리카드는..."
- **최신 사용자 질문**: "결제일자도 알려줘"
- **결과**:
{{
  "is_new_topic": false,
  "action": "DIRECT_ANSWER",
  "classification": "USER_INFO_ONLY",
  "refined_query": "고객님의 우리카드 결제일은 매월 4일입니다.",
  "reasoning": "질문에 대한 답변이 제공된 사용자 정보 안에 모두 포함되어 있으므로 USER_INFO_ONLY로 분류하고, 사용자 정보를 바탕으로 완전한 답변을 생성했습니다. '신용카드' 타입은 답변에 필수적이지 않으므로 포함하지 않았습니다."
}}

**# 예시 4: 키워드 기반 치환 - 결제일자 (우리카드만)**
- **사용자 정보**: {{
    ...
    "data": {{
      ...
      "ownCardArr": [
        {{ "bank": "우리카드", "type": "신용카드", "paymentDate": "4" }}
      ]
    }}
  }}
- **대화 기록**: "사용자: 안녕하세요 / 어시스턴트: 안녕하세요!"
- **최신 사용자 질문**: "내 카드 결제일자는 언제인가요?"
- **결과**:
{{
  "is_new_topic": true,
  "action": "SEARCH",
  "classification": "HYBRID",
  "refined_query": "우리카드 결제일자",
  "reasoning": "'내 카드'와 '결제일자' 키워드가 있어 HYBRID로 분류하고, '결제일자' 키워드 매칭에 따라 우리카드만 치환했습니다."
}}

**# 예시 5: 키워드 기반 치환 - 혜택 (신용카드만)**
- **사용자 정보**: 위와 동일
- **최신 사용자 질문**: "내 카드 혜택이 뭐가 있나요?"
- **결과**:
{{
  "is_new_topic": true,
  "action": "SEARCH",
  "classification": "HYBRID",
  "refined_query": "신용카드 혜택",
  "reasoning": "'내 카드'와 '혜택' 키워드가 있어 HYBRID로 분류하고, '혜택' 키워드 매칭에 따라 신용카드만 치환했습니다."
}}

**# 예시 6: 키워드 기반 치환 - 소득공제 (우리카드 + 신용카드)**
- **사용자 정보**: 위와 동일
- **최신 사용자 질문**: "내 카드로 소득공제 받을 수 있나요?"
- **결과**:
{{
  "is_new_topic": true,
  "action": "SEARCH",
  "classification": "HYBRID",
  "refined_query": "우리카드 신용카드 소득공제",
  "reasoning": "'내 카드'와 '소득공제' 키워드가 있어 HYBRID로 분류하고, '소득공제' 키워드 매칭에 따라 우리카드와 신용카드 모두 치환했습니다."
}}

---

### [입력 데이터]
**[사용자 정보]**
{userinfo}

**[대화 기록]**
{context}

**[최신 사용자 질문]**
{question}

---

### [결과]
**응답:**
"""

# prompt_for_query = """
# 다음 '문서 내용'을 바탕으로 질문에 답변해주세요.
# 답변은 한국어(한글)로 작성해주세요.
# 절대로 영어로 답변하지 마세요.
# 답변에 적절한 줄바꿈을 적용해주세요.
# 답변을 표로작성가능하면 표로 작성해주세요.
# 표로 작성 불가능한 경우에는 작성할 필요가 없습니다.
# 문서에서 답을 찾을 수 없다면 "문서에 관련 내용이 없습니다"라고만 답변하세요.


# 문서 내용: {context}

# 질문: {question}

# 답변:
# """

prompt_for_query = """
당신은 제공된 문서만을 기반으로 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

**[핵심 규칙 - 절대 준수]**
1. **문서 기반 답변:** 반드시 아래 제공된 문서 내용만을 사용하여 답변해야 합니다.
2. **외부 지식 금지:** 문서에 없는 내용이나 당신의 사전 지식은 절대 추가하지 마세요.
3. **정확한 매칭 원칙:** 질문에서 요구하는 **정확한 값, 조건, 항목**이 문서에 명시되어 있을 때만 답변하세요.
4. **추론 및 추정 금지:** 문서에 유사한 정보가 있어도, 질문과 **정확히 일치하지 않으면** 절대 추론하거나 추정하지 마세요.
5. **정보 부재 시 답변:** 질문의 정확한 조건이나 값이 문서에 없으면, "죄송합니다. [구체적인 조건/값]에 대한 정보를 찾을 수 없습니다."라고 명확하게 답변하세요.

**[결제일 유효성 검증 규칙]**
1. **결제일 범위 제한:** 카드 결제일은 **1일부터 27일까지만** 존재합니다.
2. **유효하지 않은 결제일 처리:**
   - 질문에서 0일, 28일, 29일, 30일, 31일 등의 결제일을 요구하면 즉시 다음과 같이 답변하세요:
   - "죄송합니다. 결제일 [X일]은 존재하지 않습니다."
3. **예시:**
   - ✅ "BC카드 결제일 15일" → 유효한 결제일이므로 문서에서 검색
   - ❌ "우리카드 결제일 28일" → "죄송합니다. 결제일 28일은 존재하지 않습니다."
   - ❌ "우리카드 결제일 29일" → "죄송합니다. 결제일 29일은 존재하지 않습니다."
   - ❌ "BC카드 결제일 30일" → "죄송합니다. 결제일 30일은 존재하지 않습니다."
   - ❌ "우리카드 결제일 31일" → "죄송합니다. 결제일 31일은 존재하지 않습니다."

**[테이블 데이터 처리 절대 규칙 - 매우 중요]**
1. **컬럼 구분 엄격 준수:** 테이블에서 각 컬럼은 명확히 구분됩니다. 절대로 다른 컬럼의 값을 혼동하지 마세요.
   - 예시 테이블 구조: | 카드사명 | 결제일 | 일시불/할부 이용기간 | 현금서비스 이용기간 |
   - **결제일 컬럼 = 두 번째 컬럼만 해당**: "1일", "5일", "8일", "12일", "13일", "15일", "23일", "25일", "27일"
   - **이용기간 컬럼 = 세 번째, 네 번째 컬럼**: "전전월 XX일 ~ 전월 XX일", "전월 XX일 ~ 당월 XX일" 등

2. **결제일 질문 처리 절대 규칙:**
   - 질문: "결제일 14일" → **반드시 결제일 컬럼(두 번째 컬럼)에서만 "14일" 검색**
   - ❌ **절대 금지**: 이용기간 컬럼의 "당월 14일", "전월 14일" 등을 결제일로 착각하지 마세요
   - ❌ **절대 금지**: "당월 14일"이 포함된 행의 다른 정보를 결제일 14일의 답변으로 사용하지 마세요

3. **정확한 행 매칭 원칙:**
   - 결제일 질문 시, 해당 결제일이 **결제일 컬럼에 정확히 존재하는 행**에서만 정보를 가져오세요
   - 존재하지 않는 결제일이면 반드시 "해당 결제일 정보가 없습니다"라고 답변하세요

4. **이용기간 답변 규칙:**
   - **완전한 정보 제공**: 결제일 질문 시, 질문에서 특정 이용기간 타입(일시불/할부 또는 현금서비스)을 명시하지 않았다면 **두 이용기간을 모두** 제공하세요.
   - **답변 형식**:
     ```
     [카드사명] 결제일 [X일]의 이용기간은 다음과 같습니다:
     - 일시불/할부 이용기간: [기간]
     - 현금서비스 이용기간: [기간]
     ```
   - **예시**:
     * 질문: "BNK 부산은행 결제일 5일 이용기간"
     * 답변: "BNK 부산은행 결제일 5일의 이용기간은 다음과 같습니다:
              - 일시불/할부 이용기간: 전전월 23일 ~ 전월 22일
              - 현금서비스 이용기간: 전전월 7일 ~ 전월 6일"
   - **특정 타입 질문**: 질문에서 "일시불", "할부", "현금서비스" 등을 명시한 경우에만 해당 타입의 이용기간만 제공하세요.

5. **컬럼 혼동 방지 예시:**
   - ✅ **올바른 매칭**: 
     * 질문 "BNK 부산은행 결제일 5일" → 결제일 컬럼에 "5일" 존재 → 해당 행의 **모든 이용기간** 제공
   - ❌ **잘못된 매칭 - 절대 금지**: 
     * 질문 "BNK 부산은행 결제일 30일" → 결제일 컬럼에 "30일" 없음 → "BNK 부산은행 결제일 30일에 대한 정보를 찾을 수 없습니다."

**[정확한 매칭 예시 - 특히 중요]**
- ✅ **정확한 매칭**: 질문 "BNK 경남은행 결제일 5일" → 문서에 "BNK 경남은행 | 5일" 존재 → 답변 가능
- ❌ **부정확한 매칭**: 질문 "BNK 경남은행 결제일 18일" → 문서에 "1일, 5일, 8일, 12일, 15일, 23일, 25일, 27일"만 존재 → "BNK 경남은행 결제일 18일에 대한 정보를 문서에서 찾을 수 없습니다." 답변
- ❌ **추론 금지**: "18일이니까 15일과 23일 사이"라는 식의 추론이나 보간 절대 금지
- ❌ **유사값 대체 금지**: 문서에 없는 값을 가장 가까운 값으로 대체하여 답변하는 것 절대 금지

**[표 데이터 처리 특별 규칙 - 강화]**
1. **정확한 행/열 매칭:** 표에서 질문의 조건과 **문자 그대로 정확히 일치**하는 행이나 열이 있을 때만 해당 정보를 제공하세요.
2. **부분 매칭 금지:** 표에 유사한 값이 있어도, 질문의 정확한 조건과 일치하지 않으면 "해당 [구체적인 값/조건]에 대한 정보가 표에 없습니다"라고 답변하세요.
3. **보간 추정 금지:** 표의 값들 사이를 계산하거나 추정하여 답변하는 것을 절대 금지합니다.
4. **존재하지 않는 데이터 처리:** 질문에서 요구하는 정확한 값(예: 결제일 18일)이 표에 없으면, 반드시 "해당 정보가 없다"고 명시해야 합니다.

**[답변 방식]**
1.  **답변 형식:**
    - 답변은 반드시 한국어로 작성합니다.
    - 내용을 효과적으로 전달할 수 있다면 표(테이블)를 적극적으로 활용하고, 그렇지 않다면 서술형으로 답변합니다.
    - 가독성을 높이기 위해 답변에 적절한 줄 바꿈을 사용합니다.
    - **문서 언급 금지:** "문서에 따르면", "제공된 문서에서", "문서 내용을 보면" 등 문서를 직접적으로 언급하는 표현은 사용하지 마세요.
    - **자연스러운 답변:** 마치 해당 정보를 직접 알고 있는 것처럼 자연스럽고 직접적으로 답변하세요.

2.  **포괄적 질문 대응:**
    - 사용자의 질문이 다소 포괄적이더라도 일반적인 답변이 가능한 경우, 먼저 문서에 기반해 답변합니다.
    - 그 후, 사용자가 더 궁금해할 만한 내용을 문서 기반으로 예측하여 2~3개의 구체적인 후속 질문을 제안합니다.
3.  **모호한 질문 대응:**
    - 질문이 너무 광범위하거나 핵심 정보가 부족하여 답변이 어려울 경우, '재질의 요청 규칙'에 따라 질문을 구체화하도록 유도합니다.

**[금지 표현 목록]**
다음과 같은 표현들은 절대 사용하지 마세요:
- "문서에 따르면"
- "제공된 문서에서"
- "문서 내용을 보면"
- "위 문서에서"
- "해당 문서에 명시된"
- "문서에 기재된"
- "자료에 따르면"
- "문서상으로는"

**[권장 답변 시작 방식]**
- 직접적 답변: "SC제일은행 결제일 5일의 이용기간은..."
- 정보 제공: "결제일에 따른 이용기간은 다음과 같습니다."
- 표 형태: "각 결제일별 이용기간은 아래와 같습니다."

**[재질의 요청 규칙 및 형식]**
- 이 규칙은 질문이 너무 일반적이거나 여러 서비스에 걸쳐 있을 때 사용합니다.
- **[중요] 질문 생성 규칙:**
    1.  **정보 소스 제한:** 아래 형식에 들어갈 질문 예시 3개는 **반드시** `[제공된 문서 내용]`만을 기반으로 생성해야 합니다.
    2.  **외부 정보 차단:** 원래 `[질문]`에 포함된 특정 키워드, 회사명, 상품명 등을 질문 예시 생성에 **절대로** 사용해서는 안 됩니다.
- **형식:**
"더 정확한 답변을 위해 다음 중 어떤 점에 대해 구체적으로 알고 싶으신지 선택해 주세요:
1. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 1]
2. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 2]
3. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 3]"

**[제공된 문서 내용]**
{context}

**[질문]**
{question}

**[답변]**
"""



prompt_for_context_summary = """
다음 '내용'을 200자 이내로 요약하여 답변하세요.
답변은 한국어(한글)로 작성해주세요.
절대로 영어로 답변하지 마세요.


내용: {context}

답변:
"""

def create_llm_chain(llm_model, prompt_template, input_variables=None):
    """
    LLMChain 생성 함수 (예외처리 포함)
    :param llm_model: 사용할 LLM 모델
    :param prompt_template: 프롬프트 템플릿
    :param input_variables: 프롬프트에 사용할 변수 리스트 (예: ["context", "question", "history"])
    :return: LLMChain 객체 또는 None

    @Description: LLM + PromptTemplate을 결합하여 실행 가능한 체인 생성
    - LLM 모델과 프롬프트를 안전하게 결합
    - 예외 처리 포함
    - 재사용 가능한 LLMChain 객체 반환
    """
    try:
        if input_variables is None:
            input_variables = ["context", "question"]
        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=input_variables
            )
        except Exception as e:
            print(f"❌ PromptTemplate 생성 오류: {str(e)}")
            return None
        try:
            chain = LLMChain(
                llm=llm_model,
                prompt=prompt
            )
        except Exception as e:
            print(f"❌ LLMChain 생성 오류: {str(e)}")
            return None
        return chain
    except Exception as e:
        print(f"❌ create_llm_chain 전체 오류: {str(e)}")
        return None

def create_retriever(embedding_model, top_k=3):
    """
    기본 Elasticsearch 기반 Retriever 생성 함수 (하위 호환성)
    :param embedding_model: 임베딩 모델 객체
    :param top_k: 검색할 문서 개수
    :return: Retriever 객체 또는 None
    """
    try:
        from core.config import ELASTICSEARCH_URL, INDEX_NAME
        from langchain_community.vectorstores import ElasticsearchStore
        es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
        if not success:
            print(f"❌ Elasticsearch 연결 실패: {message}")
            return None
        if not es_client.indices.exists(index=INDEX_NAME):
            print(f"❌ 인덱스 '{INDEX_NAME}'가 존재하지 않습니다. PDF 파일을 먼저 인덱싱하세요.")
            return None
        doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
        if doc_count == 0:
            print(f"❌ 인덱스 '{INDEX_NAME}'에 문서가 없습니다. PDF 파일을 먼저 인덱싱하세요.")
            return None
        vectorstore = ElasticsearchStore(
            embedding=embedding_model,
            index_name=INDEX_NAME,
            es_url=ELASTICSEARCH_URL,
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": top_k,
                "fetch_k": min(top_k * 3, 10000)
            }
        )
        print("✅ 기본 Retriever 생성 성공")
        return retriever
    except Exception as e:
        print(f"❌ Retriever 생성 오류: {str(e)}")
        return None

def create_enhanced_retriever(embedding_model, top_k=3):
    """
    고도화된 하이브리드 검색 Retriever 생성 함수
    - 시맨틱 + 키워드 검색 조합
    - 정교한 스코어링 시스템
    - 메타데이터 기반 가중치
    - 관련성 임계값으로 환각 방지
    
    :param embedding_model: 임베딩 모델 객체
    :param top_k: 검색할 문서 개수
    :return: Enhanced Retriever 객체 또는 None
    """
    try:
        from core.config import ELASTICSEARCH_URL, INDEX_NAME
        from langchain_community.vectorstores import ElasticsearchStore
        
        # Elasticsearch 연결 확인
        es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
        if not success:
            print(f"❌ Elasticsearch 연결 실패: {message}")
            return None
            
        if not es_client.indices.exists(index=INDEX_NAME):
            print(f"❌ 인덱스 '{INDEX_NAME}'가 존재하지 않습니다.")
            return None
            
        doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
        if doc_count == 0:
            print(f"❌ 인덱스 '{INDEX_NAME}'에 문서가 없습니다.")
            return None
        
        # 벡터스토어 생성
        vectorstore = ElasticsearchStore(
            embedding=embedding_model,
            index_name=INDEX_NAME,
            es_url=ELASTICSEARCH_URL,
        )
        
        # 기본 시맨틱 검색 리트리버 (확장된 후보군)
        base_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": top_k * 2,  # 더 많은 후보 확보
                "fetch_k": min(top_k * 6, 10000)  # 초기 검색 범위 확대
            }
        )
        
        def enhanced_retrieve(query):
            """고도화된 하이브리드 검색 로직"""
            try:
                # 1차: 시맨틱 검색
                semantic_docs = base_retriever.get_relevant_documents(query)
                
                # 2차: 키워드 검색으로 보완
                keyword_results = ElasticsearchManager.keyword_search(query, top_k * 2)
                
                # 3차: 정교한 스코어링 시스템
                query_keywords = query.lower().split()
                scored_docs = []
                
                for doc in semantic_docs:
                    content = doc.page_content.lower()

                    # 🎯 BC카드 특별 매칭 점수
                    bc_bonus = 0
                    if 'bc카드' in query.lower() or 'bc 카드' in query.lower():
                        # BC카드 질문일 때 BC바로카드 문서에 보너스 점수
                        if any(bc_term in content for bc_term in ['bc바로카드', 'bc 바로카드', 'bc카드']):
                            bc_bonus = 2
                            print(f"  ✅ BC카드 특별 매칭 보너스: +{bc_bonus}")
                    
                    # 기본 키워드 매칭 점수
                    keyword_score = sum(1 for keyword in query_keywords if keyword in content)
                    
                    # 관련성 임계값 설정 (환각 방지)
                    min_relevance_score = 0.75
                    if keyword_score == 0 and len(doc.page_content) < 50:
                        # 키워드 매칭이 전혀 없고 내용이 너무 짧으면 제외
                        continue
                    
                    metadata_score = 0
                    
                    # 메타데이터 기반 스코어링
                    if hasattr(doc, 'metadata') and doc.metadata:

                        # 표 포함 문서 가중치
                        if doc.metadata.get('type') and ('표' in query or '목록' in query or '기준' in query):
                            metadata_score += 0.5
                            
                        # 카테고리 기반 가중치
                        if doc.metadata.get('category') == 'DOCX' and ('안내' in query or '절차' in query):
                            metadata_score += 0.5
                        if doc.metadata.get('category') == 'PDF' and ('규정' in query or '정책' in query):
                            metadata_score += 0.5
                        
                        # 새로 추가: 파일명 기반 가중치 (키워드 매칭)
                        filename = doc.metadata.get('filename', '').lower()
                        
                        # 신용카드 관련 파일명 가중치
                        if '신용카드' in query.lower() and '신용카드' in filename:
                            metadata_score += 1
                            print(f"  ✅ 신용카드 파일명 매칭 보너스: {filename}")

                        # 키워드 필드 매칭 가중치
                        keywords_field = doc.metadata.get('keywords', '').lower()
                        for keyword in query_keywords:
                            if keyword in keywords_field:
                                metadata_score += 1
                    
                    # 키워드 검색 결과와 매칭 보너스
                    filename = doc.metadata.get('filename', '') if hasattr(doc, 'metadata') else ''
                    for kw_result in keyword_results:
                        if isinstance(kw_result, dict):
                            kw_filename = kw_result.get('metadata', {}).get('filename', '')
                        else:
                            kw_filename = getattr(kw_result, 'metadata', {}).get('filename', '') if hasattr(kw_result, 'metadata') else ''
                        
                        if kw_filename and kw_filename == filename:
                            metadata_score += 1
                            break
                    
                    # 🎯 BC카드 보너스 점수 포함
                    total_score = keyword_score + metadata_score + bc_bonus
                    
                    # 최소 점수 이상인 문서만 포함 (환각 방지)
                    if total_score >= min_relevance_score:
                        scored_docs.append((doc, total_score))
                
                # 스코어 기반 정렬 후 상위 k개 반환
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                final_docs = [doc for doc, _ in scored_docs[:top_k]]

                print(f"\n🏆 최종 선택된 문서들:")
                for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
                    # 🎯 content의 앞 30자만 추출
                    if hasattr(doc, 'page_content'):
                        content_preview = doc.page_content[:30] + "..." if len(doc.page_content) > 30 else doc.page_content
                    else:
                        content_preview = 'Unknown'
                    
                    print(f"   {i}. {content_preview} (점수: {score})")
                # 🎯 결제일/이용기간 관련 질문 특별 처리
                payment_keywords = ['결제일', '결제일자', '이용기간']
                is_payment_related = any(keyword in query for keyword in payment_keywords)
                
                if is_payment_related and final_docs:
                    print(f"🎯 결제일/이용기간 관련 질문 감지: {query}")
                    
                    # 첫 번째 문서가 테이블 타입이고 결제일/이용기간 정보를 포함하는지 확인
                    first_doc = final_docs[0]
                    if (hasattr(first_doc, 'metadata') and 
                        first_doc.metadata and 
                        first_doc.metadata.get('type') == 'table'):
                        
                        content = first_doc.page_content.lower()
                        # 결제일/이용기간 관련 테이블 내용인지 확인
                        table_indicators = ['결제일', '이용기간', '카드사명', '일시불', '할부', '현금서비스']
                        
                        if any(indicator in content for indicator in table_indicators):
                            print(f"✅ 결제일/이용기간 테이블 문서 발견, 첫 번째 문서만 사용")
                            print(f"   📄 파일명: {first_doc.metadata.get('filename', 'Unknown')}")
                            print(f"   📊 테이블 컬럼: {first_doc.metadata.get('column', [])}")
                            print(f"   📃 페이지: {first_doc.metadata.get('page', 'Unknown')}")
                            
                            # 첫 번째 문서만 반환
                            return [first_doc]
                    
                    # 첫 번째 문서가 적절하지 않다면 전체 문서에서 검색
                    for doc in final_docs:
                        if (hasattr(doc, 'metadata') and 
                            doc.metadata and 
                            doc.metadata.get('type') == 'table'):
                            
                            content = doc.page_content.lower()
                            table_indicators = ['결제일', '이용기간', '카드사명', '일시불', '할부', '현금서비스']
                            
                            if any(indicator in content for indicator in table_indicators):
                                print(f"✅ 결제일/이용기간 테이블 문서 발견, 해당 문서만 사용")
                                print(f"   📄 파일명: {doc.metadata.get('filename', 'Unknown')}")
                                print(f"   📊 테이블 컬럼: {doc.metadata.get('column', [])}")
                                print(f"   📃 페이지: {doc.metadata.get('page', 'Unknown')}")
                                
                                # 해당 테이블 문서만 반환
                                return [doc]
                    
                    print("⚠️ 적절한 결제일/이용기간 테이블을 찾지 못했습니다. 기존 검색 결과를 사용합니다.")

                # 빈 결과 처리
                if not final_docs:
                    print("⚠️ 관련성 있는 문서를 찾을 수 없습니다.")
                
                return final_docs
                
            except Exception as e:
                print(f"❌ 고도화된 검색 중 오류: {str(e)}")
                # 오류 시 기본 시맨틱 검색으로 폴백
                return base_retriever.get_relevant_documents(query)[:top_k]
        
        # 커스텀 리트리버 클래스 생성
        class EnhancedRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager=None):
                return enhanced_retrieve(query)
                
            def get_relevant_documents(self, query):
                return enhanced_retrieve(query)
        
        enhanced_retriever = EnhancedRetriever()
        print(f"✅ 고도화된 하이브리드 리트리버 생성 완료 (top_k: {top_k})")
        return enhanced_retriever
        
    except Exception as e:
        print(f"❌ 고도화된 Retriever 생성 오류: {str(e)}")
        return None