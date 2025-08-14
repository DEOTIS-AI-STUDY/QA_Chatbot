import streamlit as st
import os
import time
import psutil
import uuid
import urllib3
import argparse
from datetime import datetime
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# --- Langsmith 추적을 위한 라이브러리 ---
try:
    from langsmith import Client
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# --- 임베딩 및 LLM 라이브러리 ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"HuggingFace 임베딩 라이브러리 로딩 실패: {e}")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
    except ImportError as e2:
        print(f"Community HuggingFace 임베딩 라이브러리도 로딩 실패: {e2}")
        HuggingFaceEmbeddings = None
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = False

# 조건부 import - 오류 발생시 None으로 설정
try:
    from langchain_upstage import ChatUpstage
    UPSTAGE_AVAILABLE = True
except ImportError as e:
    print(f"Upstage 라이브러리 로딩 실패: {e}")
    ChatUpstage = None
    UPSTAGE_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Ollama 라이브러리 로딩 실패: {e}")
    ChatOllama = None
    OLLAMA_AVAILABLE = False

# --- 문서 처리 및 벡터 스토어 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch

# --- LangChain 체인 관련 ---
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===== 설정 =====
# Elasticsearch 설정
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "unified_rag")
PDF_DIR = os.getenv("PDF_DIR", "pdf")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# BGE-M3 임베딩 모델 설정
BGE_MODEL_NAME = "BAAI/bge-m3"

# LLM 모델 설정
LLM_MODELS = {
    "upstage": {
        "name": "Upstage Solar LLM",
        "model_id": "solar-1-mini-chat",
        "api_key_env": "UPSTAGE_API_KEY"
    },
    "qwen2": {
        "name": "Qwen2",
        "model_id": "qwen2:7b",
        "api_key_env": None
    },
    "llama3": {
        "name": "Llama3",
        "model_id": "llama3:8b",
        "api_key_env": None
    }
}

# ===== Langsmith 설정 함수 =====
def setup_langsmith():
    """Langsmith 추적 설정"""
    if not LANGSMITH_AVAILABLE:
        return None, False
        
    try:
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "unified-rag-system")
        langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
            
            tracer = LangChainTracer(project_name=langsmith_project)
            callback_manager = CallbackManager([tracer])
            return callback_manager, True
        else:
            return None, False
    except Exception as e:
        st.warning(f"⚠️ Langsmith 설정 중 오류: {str(e)}")
        return None, False

# ===== 성능 측정 클래스 =====
class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
        self.current_process = psutil.Process()
        
    def get_elasticsearch_memory(self):
        """Elasticsearch 프로세스 메모리 사용량"""
        es_memory = 0
        es_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'elasticsearch' in proc.info['name'].lower() or 'java' in proc.info['name'].lower():
                    # Java 프로세스 중 Elasticsearch 관련 확인
                    try:
                        cmdline = proc.cmdline()
                        if any('elasticsearch' in cmd.lower() for cmd in cmdline):
                            es_processes.append(proc)
                            es_memory += proc.memory_info().rss / 1024 / 1024
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return es_memory, len(es_processes)
    
    def get_ollama_memory(self):
        """Ollama 프로세스 메모리 사용량"""
        ollama_memory = 0
        ollama_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
                    ollama_memory += proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return ollama_memory, len(ollama_processes)
    
    def get_total_memory_usage(self):
        """전체 시스템 메모리 사용량"""
        python_memory = self.current_process.memory_info().rss / 1024 / 1024
        es_memory, es_count = self.get_elasticsearch_memory()
        ollama_memory, ollama_count = self.get_ollama_memory()
        
        return {
            'python_memory': python_memory,
            'elasticsearch_memory': es_memory,
            'ollama_memory': ollama_memory,
            'total_memory': python_memory + es_memory + ollama_memory,
            'elasticsearch_process_count': es_count,
            'ollama_process_count': ollama_count
        }
        
    def start_timer(self, task_name):
        """작업 시작 시간 기록"""
        memory_info = self.get_total_memory_usage()
        self.metrics[task_name] = {
            'start_time': time.time(),
            'start_python_memory': memory_info['python_memory'],
            'start_elasticsearch_memory': memory_info['elasticsearch_memory'],
            'start_ollama_memory': memory_info['ollama_memory'],
            'start_total_memory': memory_info['total_memory'],
            'start_cpu_percent': self.current_process.cpu_percent()
        }
        
    def end_timer(self, task_name):
        """작업 종료 시간 기록 및 결과 반환"""
        if task_name in self.metrics:
            end_time = time.time()
            memory_info = self.get_total_memory_usage()
            
            self.metrics[task_name].update({
                'end_time': end_time,
                'end_python_memory': memory_info['python_memory'],
                'end_elasticsearch_memory': memory_info['elasticsearch_memory'],
                'end_ollama_memory': memory_info['ollama_memory'],
                'end_total_memory': memory_info['total_memory'],
                'duration': end_time - self.metrics[task_name]['start_time'],
                'python_memory_used': memory_info['python_memory'] - self.metrics[task_name]['start_python_memory'],
                'elasticsearch_memory_used': memory_info['elasticsearch_memory'] - self.metrics[task_name]['start_elasticsearch_memory'],
                'ollama_memory_used': memory_info['ollama_memory'] - self.metrics[task_name]['start_ollama_memory'],
                'total_memory_used': memory_info['total_memory'] - self.metrics[task_name]['start_total_memory'],
                'elasticsearch_process_count': memory_info['elasticsearch_process_count'],
                'ollama_process_count': memory_info['ollama_process_count']
            })
            
            return self.metrics[task_name]
        return None
    
    def get_summary(self):
        """전체 성능 요약 반환"""
        summary = {}
        for task, metrics in self.metrics.items():
            if 'duration' in metrics:
                summary[task] = {
                    '실행시간 (초)': round(metrics['duration'], 2),
                    'Python 메모리 (MB)': round(metrics['python_memory_used'], 2),
                    'Elasticsearch 메모리 (MB)': round(metrics['elasticsearch_memory_used'], 2),
                    'Ollama 메모리 (MB)': round(metrics['ollama_memory_used'], 2),
                    '총 메모리 (MB)': round(metrics['total_memory_used'], 2),
                    'ES 프로세스 수': metrics['elasticsearch_process_count'],
                    'Ollama 프로세스 수': metrics['ollama_process_count'],
                    '완료시간': datetime.fromtimestamp(metrics['end_time']).strftime('%H:%M:%S')
                }
        return summary

# ===== 하이브리드 성능 추적 클래스 =====
class HybridPerformanceTracker:
    def __init__(self):
        self.system_tracker = PerformanceTracker()
        self.langsmith_callback, self.langsmith_enabled = setup_langsmith()
        self.hybrid_metrics = {
            'system_metrics': {},
            'langsmith_sessions': [],
            'combined_insights': {}
        }
        
    def get_langsmith_status(self):
        """Langsmith 상태 반환"""
        return {
            'available': LANGSMITH_AVAILABLE,
            'enabled': self.langsmith_enabled,
            'project': os.getenv("LANGSMITH_PROJECT", "unified-rag-system")
        }
    
    def track_preprocessing_stage(self, stage_name):
        """전처리 단계 추적"""
        return self.system_tracker.start_timer(stage_name)
    
    def end_preprocessing_stage(self, stage_name):
        """전처리 단계 종료"""
        metrics = self.system_tracker.end_timer(stage_name)
        if metrics:
            self.hybrid_metrics['system_metrics'][stage_name] = metrics
        return metrics
    
    def track_llm_inference(self, qa_chain, query, metadata=None):
        """LLM 추론 하이브리드 추적"""
        self.system_tracker.start_timer("LLM_추론_시스템")
        
        if metadata is None:
            metadata = {}
        
        enhanced_metadata = {
            **metadata,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'system': 'unified_rag'
        }
        
        try:
            if self.langsmith_enabled and self.langsmith_callback:
                response = qa_chain(
                    {"query": query}, 
                    callbacks=self.langsmith_callback.handlers,
                    metadata=enhanced_metadata
                )
            else:
                response = qa_chain({"query": query})
            
            system_metrics = self.system_tracker.end_timer("LLM_추론_시스템")
            
            combined_result = {
                'response': response,
                'system_metrics': system_metrics,
                'langsmith_enabled': self.langsmith_enabled,
                'metadata': enhanced_metadata
            }
            
            return combined_result
            
        except Exception as e:
            self.system_tracker.end_timer("LLM_추론_시스템")
            st.error(f"LLM 추론 중 오류: {str(e)}")
            raise
    
    def get_system_summary(self):
        """시스템 성능 요약"""
        return self.system_tracker.get_summary()
    
    def get_hybrid_insights(self):
        """하이브리드 성능 인사이트"""
        system_summary = self.get_system_summary()
        
        insights = {
            'system_performance': system_summary,
            'langsmith_status': self.get_langsmith_status(),
            'recommendations': self._generate_recommendations(system_summary)
        }
        
        return insights
    
    def _generate_recommendations(self, system_summary):
        """성능 기반 추천사항 생성"""
        recommendations = []
        
        if system_summary:
            slowest_task = max(system_summary.items(), 
                             key=lambda x: x[1]['실행시간 (초)'], 
                             default=(None, {'실행시간 (초)': 0}))
            
            if slowest_task[0] and slowest_task[1]['실행시간 (초)'] > 10:
                recommendations.append(f"⚠️ {slowest_task[0]}이 {slowest_task[1]['실행시간 (초)']}초로 가장 오래 걸립니다.")
                
                if 'Elasticsearch' in slowest_task[0]:
                    recommendations.append("💡 Elasticsearch 인덱스 설정을 최적화하거나 더 적은 문서로 테스트해보세요.")
                elif 'LLM_추론' in slowest_task[0]:
                    recommendations.append("💡 더 작은 LLM 모델을 선택하거나 GPU를 사용해보세요.")
            
            total_memory = sum(metrics.get('총 메모리 (MB)', 0) for metrics in system_summary.values())
            if total_memory > 8000:
                recommendations.append(f"🔥 총 메모리 사용량이 {total_memory:.1f}MB로 높습니다. 시스템 최적화를 고려해보세요.")
        
        return recommendations

# ===== 임베딩 및 LLM 모델 팩토리 =====
class ModelFactory:
    @staticmethod
    def create_embedding_model():
        """BGE-M3 임베딩 모델 생성"""
        if not HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            st.error("HuggingFace 임베딩 라이브러리를 사용할 수 없습니다.")
            return None
            
        try:
            return HuggingFaceEmbeddings(
                model_name=BGE_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"BGE-M3 임베딩 모델 로딩 실패: {str(e)}")
            # 폴백으로 기본 한국어 모델 사용
            try:
                return HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e2:
                st.error(f"폴백 임베딩 모델도 로딩 실패: {str(e2)}")
                return None
    
    @staticmethod
    def create_llm_model(model_choice):
        """선택된 LLM 모델 생성"""
        if model_choice == "upstage":
            if not UPSTAGE_AVAILABLE:
                st.error("❌ Upstage 라이브러리가 설치되지 않았습니다.")
                return None
            api_key = os.getenv("UPSTAGE_API_KEY")
            if not api_key:
                st.error("UPSTAGE_API_KEY가 설정되지 않았습니다.")
                return None
            return ChatUpstage(
                api_key=api_key,
                model=LLM_MODELS["upstage"]["model_id"],
                temperature=0
            )
        
        elif model_choice == "qwen2":
            if not OLLAMA_AVAILABLE:
                st.error("❌ Ollama 라이브러리가 설치되지 않았습니다.")
                return None
            return ChatOllama(
                model=LLM_MODELS["qwen2"]["model_id"],
                temperature=0
            )
        
        elif model_choice == "llama3":
            if not OLLAMA_AVAILABLE:
                st.error("❌ Ollama 라이브러리가 설치되지 않았습니다.")
                return None
            return ChatOllama(
                model=LLM_MODELS["llama3"]["model_id"],
                temperature=0
            )
        
        else:
            st.error(f"지원하지 않는 모델: {model_choice}")
            return None

# ===== Elasticsearch 유틸리티 =====
class ElasticsearchManager:
    @staticmethod
    def check_connection():
        """Elasticsearch 연결 확인 (개선된 버전)"""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 다양한 연결 방법 시도
        connection_configs = [
            {
                "hosts": [ELASTICSEARCH_URL],
                "verify_certs": False,
                "ssl_show_warn": False,
                "request_timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            },
            {
                "hosts": ["http://localhost:9200"],
                "verify_certs": False,
                "request_timeout": 30
            },
            {
                "hosts": ["http://127.0.0.1:9200"],
                "verify_certs": False,
                "request_timeout": 30
            }
        ]
        
        for i, config in enumerate(connection_configs):
            try:
                es = Elasticsearch(**config)
                if es.ping():
                    cluster_info = es.info()
                    version = cluster_info.get('version', {}).get('number', 'Unknown')
                    return True, f"연결 성공 (v{version}) - 방법 {i+1}"
            except Exception as conn_error:
                continue
        
        return False, "모든 연결 방법 실패. Elasticsearch가 실행 중인지 확인하세요."
    
    @staticmethod
    def get_safe_elasticsearch_client():
        """안전한 Elasticsearch 클라이언트 반환"""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        configs = [
            {
                "hosts": [ELASTICSEARCH_URL],
                "verify_certs": False,
                "ssl_show_warn": False,
                "request_timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            },
            {
                "hosts": ["http://localhost:9200"],
                "verify_certs": False,
                "request_timeout": 30
            }
        ]
        
        for config in configs:
            try:
                es = Elasticsearch(**config)
                if es.ping():
                    return es, True, f"클라이언트 생성 성공: {config['hosts'][0]}"
            except Exception:
                continue
        
        return None, False, "Elasticsearch 클라이언트 생성 실패"
    
    @staticmethod
    def list_pdfs(pdf_dir: str):
        """PDF 파일 목록 반환"""
        try:
            names = os.listdir(pdf_dir)
        except FileNotFoundError:
            return []
        
        files = []
        for name in names:
            if name.lower().endswith(".pdf"):
                files.append(os.path.join(pdf_dir, name))
        return sorted(files)
    
    @staticmethod
    def index_pdfs(pdf_files, embeddings, hybrid_tracker):
        """PDF 파일들을 Elasticsearch에 인덱싱"""
        hybrid_tracker.track_preprocessing_stage("PDF_인덱싱_시작")
        
        # 기존 인덱스 삭제
        es = Elasticsearch(ELASTICSEARCH_URL)
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
        
        all_documents = []
        
        for pdf_path in pdf_files:
            hybrid_tracker.track_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
            
            # PDF 로딩
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 텍스트 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            
            # 메타데이터 보강
            for chunk in chunks:
                chunk.metadata.update({
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "category": "PDF"
                })
            
            all_documents.extend(chunks)
            hybrid_tracker.end_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            return False, "추출된 텍스트가 없습니다."
        
        # Elasticsearch에 저장
        hybrid_tracker.track_preprocessing_stage("Elasticsearch_저장")
        try:
            # 안전한 Elasticsearch 클라이언트 확인
            es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
            if not success:
                raise Exception(f"Elasticsearch 연결 실패: {message}")
            
            # 문서 저장
            ElasticsearchStore.from_documents(
                all_documents,
                embedding=embeddings,
                es_url=ELASTICSEARCH_URL,
                index_name=INDEX_NAME
            )
            
            es_client.indices.refresh(index=INDEX_NAME)
            cnt = es_client.count(index=INDEX_NAME).get("count", 0)
            
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            
            return True, f"인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            return False, f"인덱싱 오류: {str(e)}"

# ===== 핵심 RAG 시스템 =====
def create_rag_chain(embeddings, llm_model, top_k=3):
    """RAG 체인 생성"""
    try:
        st.write("🔍 Elasticsearch 클라이언트 연결 확인 중...")
        # 안전한 Elasticsearch 클라이언트 확인
        es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
        if not success:
            error_msg = f"Elasticsearch 연결 실패: {message}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        st.write(f"✅ Elasticsearch 연결 성공: {message}")
        
        st.write("🔍 인덱스 존재 여부 확인 중...")
        # 인덱스 존재 확인
        if not es_client.indices.exists(index=INDEX_NAME):
            error_msg = f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다. PDF 파일을 먼저 인덱싱하세요."
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        # 문서 개수 확인
        doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
        if doc_count == 0:
            error_msg = f"인덱스 '{INDEX_NAME}'에 문서가 없습니다. PDF 파일을 먼저 인덱싱하세요."
            st.error(f"❌ {error_msg}")
            return None, error_msg
        st.write(f"✅ 인덱스에 {doc_count}개 문서 발견")
        
        st.write("🔍 Elasticsearch 벡터스토어 생성 중...")
        # Elasticsearch 벡터스토어 연결
        try:
            # 가장 기본적인 방법으로 시도
            vectorstore = ElasticsearchStore(
                embedding=embeddings,
                index_name=INDEX_NAME,
                es_url=ELASTICSEARCH_URL
            )
            st.write("✅ 벡터스토어 생성 완료")
        except TypeError as type_error:
            # 파라미터 이름 문제인 경우
            st.write(f"⚠️ 파라미터 오류, 다른 방법 시도: {str(type_error)}")
            try:
                vectorstore = ElasticsearchStore(
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                    elasticsearch_url=ELASTICSEARCH_URL
                )
                st.write("✅ 벡터스토어 생성 완료 (elasticsearch_url 사용)")
            except Exception as vs_error2:
                error_msg = f"벡터스토어 생성 실패: {str(vs_error2)}"
                st.error(f"❌ {error_msg}")
                return None, error_msg
        except Exception as vs_error:
            error_msg = f"벡터스토어 생성 실패: {str(vs_error)}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        st.write(f"🔍 리트리버 설정 중 (top_k={top_k})...")
        # 리트리버 설정
        try:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "fetch_k": min(top_k * 3, 10000)
                }
            )
            st.write("✅ 리트리버 설정 완료")
        except Exception as ret_error:
            error_msg = f"리트리버 설정 실패: {str(ret_error)}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        st.write("🔍 프롬프트 템플릿 설정 중...")
        # 프롬프트 템플릿
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
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            st.write("✅ 프롬프트 템플릿 설정 완료")
        except Exception as prompt_error:
            error_msg = f"프롬프트 템플릿 설정 실패: {str(prompt_error)}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        st.write("🔍 QA 체인 생성 중...")
        # QA 체인 생성
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_model,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            st.write("✅ QA 체인 생성 완료")
        except Exception as qa_error:
            error_msg = f"QA 체인 생성 실패: {str(qa_error)}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        # 최종 검증
        if qa_chain is None:
            error_msg = "QA 체인이 None으로 생성되었습니다."
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        st.write("🔍 QA 체인 테스트 중...")
        # 간단한 테스트 쿼리
        try:
            test_result = qa_chain({"query": "테스트"})
            if test_result is None:
                error_msg = "QA 체인 테스트 실패: 응답이 None입니다."
                st.error(f"❌ {error_msg}")
                return None, error_msg
            st.write("✅ QA 체인 테스트 성공")
        except Exception as test_error:
            error_msg = f"QA 체인 테스트 실패: {str(test_error)}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
        
        st.write("🎉 RAG 체인 생성 완전 성공!")
        return qa_chain, True
        
    except Exception as e:
        error_msg = f"예상치 못한 오류: {str(e)} (타입: {type(e).__name__})"
        st.error(f"❌ {error_msg}")
        
        # 스택 트레이스 추가
        import traceback
        st.error("📋 상세 스택 트레이스:")
        st.code(traceback.format_exc())
        
        return None, error_msg

# ===== Streamlit UI =====
def main():
    st.set_page_config(
        page_title="통합 RAG 시스템",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🚀 통합 RAG 시스템")
    st.markdown("**BGE-M3 임베딩 + Elasticsearch + 멀티 LLM + 성능 모니터링**")
    
    # 상단 네비게이션 바 추가
    col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 1])
    

    with col_nav2:
        # 추가 네비게이션 공간 (필요시 확장 가능)
        st.empty()
    
    st.divider()
    
    # 세션 상태 초기화
    if "hybrid_tracker" not in st.session_state:
        st.session_state.hybrid_tracker = HybridPerformanceTracker()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        # Elasticsearch 연결 상태
        es_connected, es_message = ElasticsearchManager.check_connection()
        if es_connected:
            st.success(f"✅ Elasticsearch: {es_message}")
        else:
            st.error(f"❌ Elasticsearch: {es_message}")
            st.stop()
        
        # Langsmith 상태 실시간 확인
        _, langsmith_enabled = setup_langsmith()
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "unified-rag-system")
        
        if langsmith_enabled:
            st.success(f"✅ Langsmith: {langsmith_project}")
            
            # LangSmith 관리 섹션
            with st.expander("📊 LangSmith 관리(소유자만)"):
                langsmith_url = "https://smith.langchain.com"
                
                st.markdown(f"""
                **프로젝트:** `{langsmith_project}`  
                **상태:** 🟢 활성화됨
                """)
                
                col_ls1, col_ls2 = st.columns(2)
                with col_ls1:
                    if st.button("📈 대시보드", key="langsmith_dashboard"):
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={langsmith_url}">', unsafe_allow_html=True)
                        st.info(f"LangSmith 대시보드로 이동: {langsmith_url}")
                
                with col_ls2:
                    if st.button("🔗 링크 복사", key="copy_langsmith_link"):
                        st.code(langsmith_url)
                        st.success("링크가 표시되었습니다!")
                
                st.markdown("**주요 기능:**")
                st.write("• 실시간 추론 추적")
                st.write("• 성능 메트릭 분석")
                st.write("• 오류 디버깅")
                st.write("• 비용 모니터링")
        else:
            st.info("📊 Langsmith: 비활성화")
            
            with st.expander("📊 LangSmith 설정"):
                # API 키 확인
                api_key = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
                if api_key:
                    st.warning("⚠️ API 키는 설정되어 있지만 LangSmith 라이브러리를 불러올 수 없습니다.")
                    st.info("💡 `pip install langsmith` 명령으로 LangSmith를 설치하세요.")
                else:
                    st.markdown("""
                    **LangSmith 활성화 방법:**
                    1. LangSmith API 키 발급
                    2. .env 파일에 추가:
                    ```
                    LANGSMITH_API_KEY=your_api_key
                    LANGSMITH_PROJECT=your_project_name
                    ```
                    3. 애플리케이션 재시작
                    """)
                
                if st.button("🌐 LangSmith 웹사이트", key="langsmith_website"):
                    st.markdown('<meta http-equiv="refresh" content="0; url=https://smith.langchain.com">', unsafe_allow_html=True)
                    st.info("LangSmith 웹사이트: https://smith.langchain.com")
        
        st.divider()
        
        # LLM 모델 선택
        st.subheader("🤖 LLM 모델 선택")
        
        # 사용 가능한 모델 필터링
        available_models = {}
        if UPSTAGE_AVAILABLE:
            available_models["upstage"] = LLM_MODELS["upstage"]
        if OLLAMA_AVAILABLE:
            available_models["qwen2"] = LLM_MODELS["qwen2"]
            available_models["llama3"] = LLM_MODELS["llama3"]
            
        if not available_models:
            st.error("❌ 사용 가능한 LLM 모델이 없습니다.")
            st.info("💡 패키지 설치를 확인하세요: langchain-upstage, langchain-ollama")
            model_choice = None
        else:
            model_choice = st.selectbox(
                "모델을 선택하세요:",
                options=list(available_models.keys()),
                format_func=lambda x: available_models[x]["name"],
                index=0,
                key="model_selector"
            )
            
            # 모델이 변경되면 RAG 시스템 초기화 필요
            if st.session_state.selected_model != model_choice:
                st.session_state.selected_model = model_choice
                st.session_state.qa_chain = None
                st.session_state.rag_initialized = False
                
            # 모델 상태 표시
            if model_choice == "upstage":
                api_key = os.getenv("UPSTAGE_API_KEY")
                if api_key and api_key != "your_upstage_api_key":
                    st.success("✅ Upstage API 키 설정됨")
                else:
                    st.warning("⚠️ UPSTAGE_API_KEY 환경변수가 필요합니다")
            elif model_choice in ["qwen2", "llama3"]:
                st.info("ℹ️ Ollama 서버가 실행 중인지 확인하세요")
        
        # 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("검색할 문서 수", 1, 10, 3)
        
        # 성능 모니터링 토글
        show_performance = st.checkbox("📊 성능 모니터링", value=True)
        
        # 디버깅 정보 (개발 중에만 표시)
        with st.expander("🔧 디버그 정보"):
            st.write(f"qa_chain: {st.session_state.qa_chain}")
            st.write(f"qa_chain type: {type(st.session_state.qa_chain)}")
            st.write(f"qa_chain is None: {st.session_state.qa_chain is None}")
            st.write(f"qa_chain == False: {st.session_state.qa_chain == False}")
            st.write(f"bool(qa_chain): {bool(st.session_state.qa_chain)}")
            st.write(f"rag_initialized: {st.session_state.get('rag_initialized', False)}")
            st.write(f"selected_model: {st.session_state.get('selected_model', 'None')}")
            st.write(f"current_model_choice: {model_choice if 'model_choice' in locals() else 'None'}")
            
            # 상태 리셋 버튼
            if st.button("🔄 상태 리셋", key="reset_debug"):
                st.session_state.qa_chain = None
                st.session_state.rag_initialized = False
                st.session_state.selected_model = None
                st.success("상태가 리셋되었습니다. 다시 초기화해주세요.")
        
        st.divider()
        
        # PDF 관리
        st.subheader("📄 PDF 관리")
        
        # PDF 업로드
        uploaded_files = st.file_uploader(
            "PDF 파일 업로드",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📥 파일 업로드 및 인덱싱"):
            # PDF 디렉토리 생성
            os.makedirs(PDF_DIR, exist_ok=True)
            
            # 업로드된 파일들 저장
            uploaded_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(PDF_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                uploaded_paths.append(file_path)
            
            with st.spinner("PDF 파일 인덱싱 중..."):
                # 임베딩 모델 생성
                embeddings = ModelFactory.create_embedding_model()
                
                # PDF 인덱싱
                success, message = ElasticsearchManager.index_pdfs(
                    uploaded_paths, 
                    embeddings, 
                    st.session_state.hybrid_tracker
                )
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # 기존 PDF 파일 목록
        existing_pdfs = ElasticsearchManager.list_pdfs(PDF_DIR)
        if existing_pdfs:
            st.write("**기존 PDF 파일:**")
            for pdf in existing_pdfs:
                st.write(f"• {os.path.basename(pdf)}")
            
            if st.button("🔄 기존 파일 재인덱싱"):
                with st.spinner("기존 PDF 파일 재인덱싱 중..."):
                    embeddings = ModelFactory.create_embedding_model()
                    success, message = ElasticsearchManager.index_pdfs(
                        existing_pdfs, 
                        embeddings, 
                        st.session_state.hybrid_tracker
                    )
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 대화")
        
        # RAG 시스템 상태 표시
        if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
            st.success(f"✅ RAG 시스템이 초기화되어 있습니다. (모델: {st.session_state.selected_model})")
        elif st.session_state.rag_initialized and st.session_state.qa_chain is None:
            st.error("❌ RAG 시스템 초기화에 오류가 있습니다. 다시 초기화해주세요.")
            st.info("상태가 일치하지 않습니다. 아래 '상태 리셋' 버튼을 사용하세요.")
        else:
            st.warning("⚠️ RAG 시스템이 초기화되지 않았습니다.")
            if st.session_state.qa_chain is False:
                st.error("❌ 이전 초기화에서 오류가 발생했습니다. 다시 초기화해주세요.")
        
        # RAG 체인 초기화 버튼
        if st.button("🔧 RAG 시스템 초기화"):
            if not available_models:
                st.error("❌ 사용 가능한 LLM 모델이 없습니다.")
                return
            
            with st.spinner("RAG 시스템 초기화 중..."):
                try:
                    # 현재 선택된 모델 사용
                    current_model = st.session_state.get('selected_model', model_choice)
                    
                    # 임베딩 모델 생성
                    st.write("📝 임베딩 모델 생성 중...")
                    embeddings = ModelFactory.create_embedding_model()
                    if embeddings is None:
                        st.error("❌ 임베딩 모델 생성 실패")
                        return
                    st.write("✅ 임베딩 모델 생성 완료")
                    
                    # LLM 모델 생성
                    st.write("🤖 LLM 모델 생성 중...")
                    llm_model = ModelFactory.create_llm_model(current_model)
                    if llm_model is None:
                        st.error("❌ LLM 모델 생성 실패")
                        return
                    st.write("✅ LLM 모델 생성 완료")
                    
                    # RAG 체인 생성
                    st.write("🔗 RAG 체인 생성 중...")
                    qa_chain, success_or_error = create_rag_chain(embeddings, llm_model, top_k)
                    
                    # 결과 분석
                    st.write(f"🔍 RAG 체인 생성 결과 분석:")
                    st.write(f"   - qa_chain 값: {qa_chain}")
                    st.write(f"   - qa_chain 타입: {type(qa_chain)}")
                    st.write(f"   - success_or_error 값: {success_or_error}")
                    st.write(f"   - success_or_error 타입: {type(success_or_error)}")
                    
                    if success_or_error is True and qa_chain is not None:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.selected_model = current_model
                        st.session_state.rag_initialized = True
                        
                        # 초기화 메시지 추가
                        if not st.session_state.messages:
                            st.session_state.messages = []
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"RAG 시스템이 초기화되었습니다. 모델: {LLM_MODELS[current_model]['name']}"
                        })
                        
                        st.success("🎉 RAG 시스템 초기화 완료!")
                        st.info("이제 아래에서 질문하실 수 있습니다.")
                    else:
                        st.session_state.qa_chain = None
                        st.session_state.rag_initialized = False
                        
                        # 오류 메시지 처리
                        if success_or_error is True:
                            error_msg = "qa_chain이 None으로 반환됨 (내부 로직 오류)"
                        elif isinstance(success_or_error, str):
                            error_msg = success_or_error
                        else:
                            error_msg = f"알 수 없는 반환값: {success_or_error}"
                            
                        st.error(f"❌ RAG 시스템 초기화 실패: {error_msg}")
                        
                        # 자동 진단 시스템
                        st.info("� 자동 진단을 실행합니다...")
                        with st.expander("�📋 상세 진단 결과", expanded=True):
                            # 1. Elasticsearch 인덱스 문서 확인
                            st.write("**1. Elasticsearch 인덱스 확인**")
                            try:
                                es_client, es_success, es_msg = ElasticsearchManager.get_safe_elasticsearch_client()
                                if es_success:
                                    try:
                                        if es_client.indices.exists(index=INDEX_NAME):
                                            doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
                                            if doc_count > 0:
                                                st.success(f"✅ 인덱스 '{INDEX_NAME}'에 {doc_count}개 문서가 있습니다.")
                                            else:
                                                st.error(f"❌ 인덱스 '{INDEX_NAME}'가 비어있습니다. PDF 파일을 먼저 인덱싱하세요.")
                                                st.info("💡 사이드바에서 PDF 파일을 업로드하거나 '기존 파일 재인덱싱' 버튼을 클릭하세요.")
                                        else:
                                            st.error(f"❌ 인덱스 '{INDEX_NAME}'가 존재하지 않습니다.")
                                            st.info("💡 사이드바에서 PDF 파일을 업로드하여 인덱스를 생성하세요.")
                                    except Exception as idx_e:
                                        st.error(f"❌ 인덱스 확인 중 오류: {str(idx_e)}")
                                else:
                                    st.error(f"❌ Elasticsearch 연결 실패: {es_msg}")
                            except Exception as es_e:
                                st.error(f"❌ Elasticsearch 진단 실패: {str(es_e)}")
                            
                            st.divider()
                            
                            # 2. LLM 모델 가용성 확인
                            st.write("**2. LLM 모델 가용성 확인**")
                            try:
                                test_llm = ModelFactory.create_llm_model(current_model)
                                if test_llm is not None:
                                    # 간단한 테스트 호출
                                    try:
                                        if current_model == "upstage":
                                            # Upstage API 테스트
                                            test_response = test_llm.invoke("안녕하세요")
                                            st.success(f"✅ {LLM_MODELS[current_model]['name']} 모델이 정상 작동합니다.")
                                        else:
                                            # Ollama 모델 테스트 (간단한 ping)
                                            try:
                                                import requests
                                                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                                                if response.status_code == 200:
                                                    models = response.json().get("models", [])
                                                    model_names = [m.get("name", "") for m in models]
                                                    target_model = LLM_MODELS[current_model]["model_id"]
                                                    if any(target_model in name for name in model_names):
                                                        st.success(f"✅ Ollama에서 {target_model} 모델을 사용할 수 있습니다.")
                                                    else:
                                                        st.error(f"❌ Ollama에 {target_model} 모델이 없습니다.")
                                                        st.info(f"💡 터미널에서 'ollama pull {target_model}' 명령을 실행하세요.")
                                                        st.code(f"ollama pull {target_model}")
                                                else:
                                                    st.error("❌ Ollama 서버 응답 오류")
                                            except requests.exceptions.RequestException:
                                                st.error("❌ Ollama 서버에 연결할 수 없습니다.")
                                                st.info("💡 터미널에서 'ollama serve' 명령으로 Ollama를 시작하세요.")
                                                st.code("ollama serve")
                                    except Exception as test_e:
                                        st.warning(f"⚠️ 모델 생성은 됐지만 테스트 호출 실패: {str(test_e)}")
                                        if current_model in ["qwen2", "llama3"]:
                                            st.info("💡 Ollama 서버가 실행 중이고 모델이 다운로드되어 있는지 확인하세요.")
                                else:
                                    st.error(f"❌ {LLM_MODELS[current_model]['name']} 모델 생성 실패")
                            except Exception as llm_e:
                                st.error(f"❌ LLM 모델 진단 실패: {str(llm_e)}")
                            
                            st.divider()
                            
                            # 3. API 키 및 환경변수 확인
                            st.write("**3. API 키 및 환경변수 확인**")
                            if current_model == "upstage":
                                api_key = os.getenv("UPSTAGE_API_KEY")
                                if api_key and api_key != "your_upstage_api_key":
                                    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "****"
                                    st.success(f"✅ UPSTAGE_API_KEY 설정됨 ({masked_key})")
                                else:
                                    st.error("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
                                    st.info("💡 .env 파일에 UPSTAGE_API_KEY=your_actual_key를 추가하세요.")
                                    st.code("echo 'UPSTAGE_API_KEY=your_actual_key' >> .env")
                            else:
                                st.success(f"✅ {LLM_MODELS[current_model]['name']}는 API 키가 필요하지 않습니다.")
                            
                            # 환경변수 상태
                            st.write("**환경변수 상태:**")
                            env_vars = {
                                "ELASTICSEARCH_URL": ELASTICSEARCH_URL,
                                "INDEX_NAME": INDEX_NAME,
                                "PDF_DIR": PDF_DIR
                            }
                            for var, value in env_vars.items():
                                st.write(f"• {var}: `{value}`")
                            
                            st.divider()
                            
                            # 4. 추천 해결 방법
                            st.write("**4. 추천 해결 방법**")
                            st.info("📝 **단계별 해결 방법:**")
                            st.write("1️⃣ PDF 파일이 업로드되어 있는지 확인")
                            st.write("2️⃣ Elasticsearch가 실행 중인지 확인 (Docker Compose 사용)")
                            st.write("3️⃣ 선택한 LLM 모델 서비스가 실행 중인지 확인")
                            st.write("4️⃣ 필요한 경우 API 키가 올바르게 설정되었는지 확인")
                            st.write("5️⃣ '상태 리셋' 후 다시 초기화")
                            
                            # 빠른 액션 버튼들
                            st.write("**빠른 액션:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("🔄 상태 리셋", key="quick_reset"):
                                    st.session_state.qa_chain = None
                                    st.session_state.rag_initialized = False
                                    st.session_state.selected_model = None
                                    st.success("상태 리셋 완료!")
                            with col2:
                                if st.button("📊 ES 상태", key="check_es"):
                                    es_client, es_success, es_msg = ElasticsearchManager.get_safe_elasticsearch_client()
                                    if es_success:
                                        info = es_client.info()
                                        st.json({"status": "connected", "version": info.get("version", {})})
                                    else:
                                        st.error(es_msg)
                            with col3:
                                if st.button("🤖 모델 테스트", key="test_model"):
                                    test_model = ModelFactory.create_llm_model(current_model)
                                    if test_model:
                                        st.success(f"{current_model} 모델 생성 성공!")
                                    else:
                                        st.error(f"{current_model} 모델 생성 실패!")
                        
                except Exception as e:
                    st.session_state.qa_chain = None
                    st.session_state.rag_initialized = False
                    st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
                    
                    # 예외 상황에서도 진단 정보 제공
                    with st.expander("🔍 오류 진단 정보", expanded=True):
                        st.write("**오류 상세 정보:**")
                        st.code(str(e))
                        
                        st.write("**가능한 원인:**")
                        if "elasticsearch" in str(e).lower():
                            st.write("• Elasticsearch 연결 문제")
                            st.info("💡 Elasticsearch가 실행 중인지 확인하세요: `docker-compose up elasticsearch`")
                        elif "ollama" in str(e).lower() or "connection" in str(e).lower():
                            st.write("• Ollama 서버 연결 문제")
                            st.info("💡 Ollama 서버를 시작하세요: `ollama serve`")
                        elif "api" in str(e).lower() or "key" in str(e).lower():
                            st.write("• API 키 관련 문제")
                            st.info("💡 .env 파일의 API 키를 확인하세요")
                        else:
                            st.write("• 알 수 없는 오류")
                            st.info("💡 로그를 확인하고 필요시 시스템을 재시작하세요")
                        
                        if hasattr(e, '__traceback__'):
                            import traceback
                            with st.expander("상세 트레이스백"):
                                st.code(traceback.format_exc())
        
        # 채팅 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
                    with st.spinner("답변 생성 중..."):
                        # 현재 모델 확인
                        current_model = st.session_state.get('selected_model')
                        
                        # 하이브리드 추적으로 LLM 추론
                        if show_performance:
                            metadata = {
                                'model': LLM_MODELS[current_model]['name'] if current_model else 'Unknown',
                                'top_k': top_k,
                                'query_length': len(prompt)
                            }
                            
                            combined_result = st.session_state.hybrid_tracker.track_llm_inference(
                                st.session_state.qa_chain,
                                prompt,
                                metadata
                            )
                            
                            response = combined_result['response']
                            system_metrics = combined_result['system_metrics']
                        else:
                            response = st.session_state.qa_chain({"query": prompt})
                            system_metrics = None
                        
                        # 답변 표시
                        st.markdown(response["result"])
                        
                        # 성능 정보 표시
                        if show_performance and system_metrics:
                            with st.expander("⚡ 성능 정보"):
                                perf_col1, perf_col2, perf_col3 = st.columns(3)
                                with perf_col1:
                                    st.metric("응답 시간", f"{system_metrics['duration']:.2f}초")
                                with perf_col2:
                                    st.metric("총 메모리", f"{system_metrics['total_memory_used']:.2f}MB")
                                with perf_col3:
                                    st.metric("ES 프로세스", system_metrics['elasticsearch_process_count'])
                        
                        # 소스 문서 표시
                        with st.expander("📄 참고 문서"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.write(f"**문서 {i+1}:**")
                                st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.write(f"*출처: {doc.metadata.get('filename', 'Unknown')}*")
                                st.divider()
                else:
                    st.warning("먼저 RAG 시스템을 초기화해주세요.")
                    st.info("위의 '🔧 RAG 시스템 초기화' 버튼을 클릭하세요.")
                    if st.session_state.rag_initialized and st.session_state.qa_chain is None:
                        st.error("⚠️ 시스템 상태 불일치가 감지되었습니다. 사이드바의 '상태 리셋' 버튼을 사용하세요.")
            
            if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
    
    with col2:
        if show_performance:
            st.subheader("📈 성능 대시보드")
            
            # 현재 시스템 정보
            memory_info = st.session_state.hybrid_tracker.system_tracker.get_total_memory_usage()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            st.metric("시스템 CPU", f"{cpu_percent:.1f}%")
            st.metric("Python 메모리", f"{memory_info['python_memory']:.1f}MB")
            st.metric("Elasticsearch", f"{memory_info['elasticsearch_memory']:.1f}MB")
            st.metric("Ollama", f"{memory_info['ollama_memory']:.1f}MB")
            st.metric("총 메모리", f"{memory_info['total_memory']:.1f}MB")
            
            # 하이브리드 인사이트
            hybrid_insights = st.session_state.hybrid_tracker.get_hybrid_insights()
            
            # 성능 요약
            perf_summary = hybrid_insights['system_performance']
            if perf_summary:
                st.subheader("작업 요약")
                for task, metrics in perf_summary.items():
                    st.write(f"**{task}**: {metrics['실행시간 (초)']}초")
            
            # 추천사항
            recommendations = hybrid_insights['recommendations']
            if recommendations:
                st.subheader("🎯 최적화 추천")
                for rec in recommendations:
                    st.write(f"• {rec}")
            

if __name__ == "__main__":
    main()