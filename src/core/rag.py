"""
RAG 시스템 핵심 로직
"""
import os
from typing import Tuple, Union
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import ElasticsearchStore
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
        
        # 리트리버 설정 - 상세한 예외 처리
        try:
            print("🔄 리트리버 설정 중...")
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "fetch_k": min(top_k * 3, 10000)
                }
            )
            print(f"✅ 리트리버 설정 완료 (top_k: {top_k})")
        except AttributeError as attr_error:
            return None, f"리트리버 설정 실패 - 속성 오류: {str(attr_error)}\n벡터스토어 객체가 올바르지 않을 수 있습니다."
        except ValueError as value_error:
            return None, f"리트리버 설정 실패 - 값 오류: {str(value_error)}\nsearch_kwargs 설정을 확인하세요."
        except Exception as ret_error:
            return None, f"리트리버 설정 실패 - 예상치 못한 오류: {str(ret_error)}"
        
        # 프롬프트 템플릿 - 상세한 예외 처리
        prompt_template = """
다음 문서 내용을 바탕으로 질문에 답변해주세요.
문서에서 답을 찾을 수 없다면 "문서에 관련 내용이 없습니다"라고 답변하세요.
답변은 친절하고 자세하게 해주세요.

문서 내용:
{context}

질문: {question}

답변:
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
