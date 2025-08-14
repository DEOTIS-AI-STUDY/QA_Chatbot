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
    from langchain_upstage import ChatUpstage, UpstageEmbeddings
    UPSTAGE_AVAILABLE = True
except ImportError as e:
    print(f"Upstage 라이브러리 로딩 실패: {e}")
    ChatUpstage = None
    UpstageEmbeddings = None
    UPSTAGE_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM, ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Ollama 라이브러리 로딩 실패: {e}")
    OllamaLLM = None
    ChatOllama = None
    OLLAMA_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"OpenAI 라이브러리 로딩 실패: {e}")
    ChatOpenAI = None
    OPENAI_AVAILABLE = False

# --- 문서 처리 및 벡터 스토어 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch

# --- LangChain 체인 관련 ---
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ===== 설정 =====
# Elasticsearch 설정
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "unified_rag")
PDF_DIR = os.getenv("PDF_DIR", "pdf")

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
                index_name=INDEX_NAME,
                ssl_verify=False
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
        # 안전한 Elasticsearch 클라이언트 확인
        es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
        if not success:
            raise Exception(f"Elasticsearch 연결 실패: {message}")
        
        # Elasticsearch 벡터스토어 연결
        vectorstore = ElasticsearchStore(
            embedding=embeddings,
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            ssl_verify=False
        )
        
        # 리트리버 설정
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": top_k,
                "fetch_k": min(top_k * 3, 10000)
            }
        )
        
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
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain, True
        
    except Exception as e:
        return None, f"RAG 체인 생성 오류: {str(e)}"

# ===== Streamlit UI =====
def main():
    st.set_page_config(
        page_title="통합 RAG 시스템",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🚀 통합 RAG 시스템")
    st.markdown("**BGE-M3 임베딩 + Elasticsearch + 멀티 LLM + 성능 모니터링**")
    
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
        
        # Langsmith 상태
        langsmith_status = st.session_state.hybrid_tracker.get_langsmith_status()
        if langsmith_status['enabled']:
            st.success(f"✅ Langsmith: {langsmith_status['project']}")
        else:
            st.info("📊 Langsmith: 비활성화")
        
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
                index=0
            )
            
            # 모델 상태 표시
            if model_choice == "upstage":
                api_key = os.getenv("UPSTAGE_API_KEY")
                if api_key:
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
        if st.session_state.qa_chain is not None:
            st.success("✅ RAG 시스템이 초기화되어 있습니다.")
        else:
            st.warning("⚠️ RAG 시스템이 초기화되지 않았습니다.")
        
        # RAG 체인 초기화 버튼
        if st.button("🔧 RAG 시스템 초기화"):
            with st.spinner("RAG 시스템 초기화 중..."):
                try:
                    # 임베딩 모델 생성
                    st.write("📝 임베딩 모델 생성 중...")
                    embeddings = ModelFactory.create_embedding_model()
                    if embeddings is None:
                        st.error("❌ 임베딩 모델 생성 실패")
                        st.stop()
                    st.write("✅ 임베딩 모델 생성 완료")
                    
                    # LLM 모델 생성
                    st.write("🤖 LLM 모델 생성 중...")
                    llm_model = ModelFactory.create_llm_model(model_choice)
                    if llm_model is None:
                        st.error("❌ LLM 모델 생성 실패")
                        st.stop()
                    st.write("✅ LLM 모델 생성 완료")
                    
                    # RAG 체인 생성
                    st.write("🔗 RAG 체인 생성 중...")
                    qa_chain, success = create_rag_chain(embeddings, llm_model, top_k)
                    
                    if success:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.messages = [
                            {"role": "assistant", "content": f"RAG 시스템이 초기화되었습니다. 모델: {LLM_MODELS[model_choice]['name']}"}
                        ]
                        st.success("🎉 RAG 시스템 초기화 완료!")
                        st.rerun()  # 페이지 새로고침으로 상태 반영
                    else:
                        st.error(f"❌ RAG 시스템 초기화 실패: {qa_chain}")
                        
                except Exception as e:
                    st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
                    st.exception(e)
        
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
                if st.session_state.qa_chain is not None:
                    with st.spinner("답변 생성 중..."):
                        # 하이브리드 추적으로 LLM 추론
                        if show_performance:
                            metadata = {
                                'model': LLM_MODELS[model_choice]['name'],
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
            
            if st.session_state.qa_chain is not None:
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