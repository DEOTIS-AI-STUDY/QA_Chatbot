import streamlit as st
import os
import time
import psutil
import uuid
from datetime import datetime
from tempfile import NamedTemporaryFile

# .env 파일 로딩
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일이 있으면 자동으로 로드
except ImportError:
    pass  # python-dotenv가 없어도 계속 진행

# --- Langsmith 추적을 위한 라이브러리 ---
try:
    from langsmith import Client
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# --- 변경/추가된 라이브러리 ---
# LLM과 임베딩 모델을 로컬로 돌리기 위한 라이브러리
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
# -----------------------------

# LangChain 관련 기존 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- OpenAI API 키 설정 부분은 이제 필요 없습니다. ---

# Langsmith 설정 함수
def setup_langsmith():
    """Langsmith 추적 설정"""
    if not LANGSMITH_AVAILABLE:
        return None, False
        
    # 환경변수 또는 Streamlit secrets에서 Langsmith API 키 가져오기
    try:
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "pdf-qa-hybrid-monitoring")
        langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")  # 기본값 포함
        
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint  # .env에서 가져온 값 사용
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
            
            # Langsmith tracer 설정
            tracer = LangChainTracer(project_name=langsmith_project)
            callback_manager = CallbackManager([tracer])
            return callback_manager, True
        else:
            return None, False
    except Exception as e:
        st.warning(f"⚠️ Langsmith 설정 중 오류: {str(e)}")
        return None, False

# 하이브리드 성능 추적 클래스
class HybridPerformanceTracker:
    def __init__(self):
        # 커스텀 시스템 리소스 추적
        self.system_tracker = PerformanceTracker()
        
        # Langsmith 설정
        self.langsmith_callback, self.langsmith_enabled = setup_langsmith()
        
        # 추적 데이터 통합
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
            'project': os.getenv("LANGSMITH_PROJECT", "pdf-qa-hybrid-monitoring")
        }
    
    def track_preprocessing_stage(self, stage_name):
        """전처리 단계는 커스텀 추적 (시스템 리소스 중심)"""
        return self.system_tracker.start_timer(stage_name)
    
    def end_preprocessing_stage(self, stage_name):
        """전처리 단계 종료"""
        metrics = self.system_tracker.end_timer(stage_name)
        if metrics:
            self.hybrid_metrics['system_metrics'][stage_name] = metrics
        return metrics
    
    def track_llm_inference(self, qa_chain, query, metadata=None):
        """LLM 추론은 Langsmith + 커스텀 추적 조합"""
        # 시스템 리소스 추적 시작
        self.system_tracker.start_timer("LLM_추론_시스템")
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        
        enhanced_metadata = {
            **metadata,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'model': 'llama3:8b'
        }
        
        try:
            if self.langsmith_enabled and self.langsmith_callback:
                # Langsmith 추적과 함께 실행
                response = qa_chain(
                    {"query": query}, 
                    callbacks=self.langsmith_callback.handlers,
                    metadata=enhanced_metadata
                )
            else:
                # Langsmith 없이 실행
                response = qa_chain({"query": query})
            
            # 시스템 리소스 추적 종료
            system_metrics = self.system_tracker.end_timer("LLM_추론_시스템")
            
            # 결과 통합
            combined_result = {
                'response': response,
                'system_metrics': system_metrics,
                'langsmith_enabled': self.langsmith_enabled,
                'metadata': enhanced_metadata
            }
            
            return combined_result
            
        except Exception as e:
            # 에러 발생 시에도 시스템 추적 종료
            self.system_tracker.end_timer("LLM_추론_시스템")
            st.error(f"LLM 추론 중 오류: {str(e)}")
            raise
    
    def get_system_summary(self):
        """시스템 성능 요약 (기존 방식)"""
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
        """성능 데이터 기반 추천사항 생성"""
        recommendations = []
        
        if system_summary:
            # 가장 느린 작업 식별
            slowest_task = max(system_summary.items(), 
                             key=lambda x: x[1]['실행시간 (초)'], 
                             default=(None, {'실행시간 (초)': 0}))
            
            if slowest_task[0] and slowest_task[1]['실행시간 (초)'] > 10:
                recommendations.append(f"⚠️ {slowest_task[0]}이 {slowest_task[1]['실행시간 (초)']}초로 가장 오래 걸립니다.")
                
                if '벡터_스토어_생성' in slowest_task[0]:
                    recommendations.append("💡 문서 크기를 줄이거나 청크 크기를 조정해보세요.")
                elif 'LLM_추론' in slowest_task[0]:
                    recommendations.append("💡 더 작은 모델을 사용하거나 Ollama 설정을 최적화해보세요.")
            
            # 메모리 사용량 체크
            total_memory = sum(metrics.get('총 메모리 (MB)', 0) for metrics in system_summary.values())
            if total_memory > 8000:  # 8GB 이상
                recommendations.append(f"🔥 총 메모리 사용량이 {total_memory:.1f}MB로 높습니다. 시스템 최적화를 고려해보세요.")
        
        return recommendations

# 성능 측정 유틸리티 함수들
class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
        self.current_process = psutil.Process()  # 현재 프로세스 객체 저장
        
    def get_ollama_processes(self):
        """Ollama 관련 프로세스들을 찾아서 반환"""
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return ollama_processes
    
    def get_total_memory_usage(self):
        """현재 프로세스 + Ollama 프로세스들의 총 메모리 사용량"""
        # 현재 Python 프로세스 메모리
        python_memory = self.current_process.memory_info().rss / 1024 / 1024
        
        # Ollama 프로세스들 메모리
        ollama_memory = 0
        ollama_processes = self.get_ollama_processes()
        for proc in ollama_processes:
            try:
                ollama_memory += proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return {
            'python_memory': python_memory,
            'ollama_memory': ollama_memory,
            'total_memory': python_memory + ollama_memory,
            'ollama_process_count': len(ollama_processes)
        }
        
    def start_timer(self, task_name):
        """작업 시작 시간 기록 (개선된 버전)"""
        memory_info = self.get_total_memory_usage()
        self.metrics[task_name] = {
            'start_time': time.time(),
            'start_python_memory': memory_info['python_memory'],
            'start_ollama_memory': memory_info['ollama_memory'],
            'start_total_memory': memory_info['total_memory'],
            'start_cpu_percent': self.current_process.cpu_percent()  # 현재 프로세스의 CPU 사용률
        }
        
    def end_timer(self, task_name):
        """작업 종료 시간 기록 및 결과 반환 (개선된 버전)"""
        if task_name in self.metrics:
            end_time = time.time()
            memory_info = self.get_total_memory_usage()
            
            self.metrics[task_name].update({
                'end_time': end_time,
                'end_python_memory': memory_info['python_memory'],
                'end_ollama_memory': memory_info['ollama_memory'],
                'end_total_memory': memory_info['total_memory'],
                'duration': end_time - self.metrics[task_name]['start_time'],
                'python_memory_used': memory_info['python_memory'] - self.metrics[task_name]['start_python_memory'],
                'ollama_memory_used': memory_info['ollama_memory'] - self.metrics[task_name]['start_ollama_memory'],
                'total_memory_used': memory_info['total_memory'] - self.metrics[task_name]['start_total_memory'],
                'ollama_process_count': memory_info['ollama_process_count']
            })
            
            return self.metrics[task_name]
        return None
    
    def get_summary(self):
        """전체 성능 요약 반환 (개선된 버전)"""
        summary = {}
        for task, metrics in self.metrics.items():
            if 'duration' in metrics:
                summary[task] = {
                    '실행시간 (초)': round(metrics['duration'], 2),
                    'Python 메모리 (MB)': round(metrics['python_memory_used'], 2),
                    'Ollama 메모리 (MB)': round(metrics['ollama_memory_used'], 2),
                    '총 메모리 (MB)': round(metrics['total_memory_used'], 2),
                    'Ollama 프로세스 수': metrics['ollama_process_count'],
                    '완료시간': datetime.fromtimestamp(metrics['end_time']).strftime('%H:%M:%S')
                }
        return summary

# 전역 하이브리드 성능 추적기 초기화
if "hybrid_tracker" not in st.session_state:
    st.session_state.hybrid_tracker = HybridPerformanceTracker()

# 하위 호환성을 위한 기존 tracker 유지
if "performance_tracker" not in st.session_state:
    st.session_state.performance_tracker = st.session_state.hybrid_tracker.system_tracker

# 세션 ID 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 1. 핵심 기능 함수 정의
def process_pdf_and_create_qa_chain(pdf_file, hybrid_tracker):
    """PDF를 처리하여 QA 체인을 생성하는 함수 (하이브리드 추적)"""
    
    # 전체 처리 시간 추적 시작
    hybrid_tracker.track_preprocessing_stage("전체_PDF_처리")
    
    # 1. PDF 로딩
    hybrid_tracker.track_preprocessing_stage("PDF_로딩")
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    hybrid_tracker.end_preprocessing_stage("PDF_로딩")
    
    # 문서 정보 저장
    doc_count = len(documents)
    total_chars = sum(len(doc.page_content) for doc in documents)

    # 2. 텍스트 분할
    hybrid_tracker.track_preprocessing_stage("텍스트_분할")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    chunk_count = len(docs)
    hybrid_tracker.end_preprocessing_stage("텍스트_분할")

    # 3. 임베딩 모델 로딩
    hybrid_tracker.track_preprocessing_stage("임베딩_모델_로딩")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} # CPU 사용, GPU 사용 시 'cuda'로 변경
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    hybrid_tracker.end_preprocessing_stage("임베딩_모델_로딩")
    
    # 4. 벡터 스토어 생성 (임베딩 생성)
    hybrid_tracker.track_preprocessing_stage("벡터_스토어_생성")
    vectorstore = FAISS.from_documents(docs, embeddings)
    hybrid_tracker.end_preprocessing_stage("벡터_스토어_생성")

    # 5. QA 체인 구성
    hybrid_tracker.track_preprocessing_stage("QA_체인_구성")
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    prompt_template = """
    주어진 문서 내용만을 사용하여 다음 질문에 답변해 주세요. 문서에서 답을 찾을 수 없다면 "문서에 관련 내용이 없습니다."라고 답변하세요.

    {context}

    질문: {question}
    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # LLM 모델 로딩
    llm = ChatOllama(model="llama3:8b", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    hybrid_tracker.end_preprocessing_stage("QA_체인_구성")
    
    # 전체 처리 완료
    hybrid_tracker.end_preprocessing_stage("전체_PDF_처리")
    
    # 파일 정리
    os.remove(tmp_file_path)
    
    # 처리 통계 저장
    processing_stats = {
        '문서_페이지_수': doc_count,
        '총_문자_수': total_chars,
        '청크_수': chunk_count,
        '검색_문서_수': 3
    }
    
    return qa_chain, processing_stats

# 2. Streamlit UI 구성
st.set_page_config(page_title="로컬 PDF QA 챗봇 (Hybrid Monitoring)", page_icon="🤖")
st.title("📄 로컬 PDF 기반 QA 챗봇 (Ollama + Langsmith)")

# 성능 모니터링 토글
show_performance = st.sidebar.checkbox("🔍 성능 모니터링 활성화", value=True)

# Langsmith 상태 표시
langsmith_status = st.session_state.hybrid_tracker.get_langsmith_status()
if langsmith_status['enabled']:
    st.success(f"✅ Langsmith 추적 활성화됨 (프로젝트: {langsmith_status['project']})")
else:
    if langsmith_status['available']:
        st.info("📊 Langsmith 라이브러리 사용 가능 - API 키를 설정하면 고급 추적이 활성화됩니다.")
    else:
        st.warning("⚠️ Langsmith 라이브러리 없음 - 커스텀 성능 측정만 사용됩니다.")

# --- OpenAI API 키 입력 부분이 필요 없어졌습니다. ---
st.info("사이드바에서 PDF 파일을 업로드하고 'PDF 처리' 버튼을 눌러주세요.")

# 세션 상태 초기화 (동일)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = None

# 사이드바에 PDF 업로더 배치 (동일)
with st.sidebar:
    st.header("PDF 업로드")
    uploaded_file = st.file_uploader("질문할 PDF 파일을 업로드하세요.", type="pdf")
    if st.button("PDF 처리"):
        if uploaded_file is not None:
            with st.spinner("하이브리드 모니터링으로 PDF를 처리 중입니다... (Langsmith + 커스텀 추적)"):
                # 하이브리드 추적기 재초기화 (새로운 처리를 위해)
                st.session_state.hybrid_tracker = HybridPerformanceTracker()
                st.session_state.performance_tracker = st.session_state.hybrid_tracker.system_tracker
                
                # PDF 처리 및 QA 체인 생성 (하이브리드 추적)
                qa_chain, processing_stats = process_pdf_and_create_qa_chain(
                    uploaded_file, 
                    st.session_state.hybrid_tracker
                )
                
                st.session_state.qa_chain = qa_chain
                st.session_state.processing_stats = processing_stats
                st.session_state.messages = [{"role": "assistant", "content": "PDF 처리가 완료되었습니다. Langsmith와 커스텀 추적이 활성화된 상태입니다!"}]
            
            st.success("PDF 처리가 완료되었습니다!")
            
            # 하이브리드 성능 결과 표시
            if show_performance:
                st.subheader("📊 하이브리드 PDF 처리 성능 분석")
                
                # 처리 통계
                st.write("**문서 정보:**")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("페이지 수", f"{processing_stats['문서_페이지_수']}개")
                    st.metric("청크 수", f"{processing_stats['청크_수']}개")
                with stats_col2:
                    st.metric("총 문자 수", f"{processing_stats['총_문자_수']:,}자")
                    st.metric("검색 문서 수", f"{processing_stats['검색_문서_수']}개")
                
                # 하이브리드 인사이트 표시
                hybrid_insights = st.session_state.hybrid_tracker.get_hybrid_insights()
                
                st.write("**시스템 성능 분석:**")
                perf_summary = hybrid_insights['system_performance']
                
                for task, metrics in perf_summary.items():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"{task}", f"{metrics['실행시간 (초)']}초")
                    with col2:
                        st.metric("Python 메모리", f"{metrics['Python 메모리 (MB)']}MB")
                    with col3:
                        st.metric("Ollama 메모리", f"{metrics['Ollama 메모리 (MB)']}MB")
                    with col4:
                        st.metric("총 메모리", f"{metrics['총 메모리 (MB)']}MB")
                
                # Langsmith 상태 및 추천사항
                st.write("**추적 시스템 상태:**")
                langsmith_info = hybrid_insights['langsmith_status']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🔧 커스텀 추적: ✅ 활성화\n📊 Langsmith: {'✅ 활성화' if langsmith_info['enabled'] else '❌ 비활성화'}")
                with col2:
                    if langsmith_info['enabled']:
                        st.success(f"프로젝트: {langsmith_info['project']}")
                    else:
                        st.warning("Langsmith API 키 설정 필요")
                
                # 성능 추천사항
                recommendations = hybrid_insights['recommendations']
                if recommendations:
                    st.write("**🎯 성능 최적화 추천:**")
                    for rec in recommendations:
                        st.write(f"- {rec}")
                
        else:
            st.error("PDF 파일을 업로드해주세요.")

# 채팅 기록 표시 및 사용자 입력 처리 (동일)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 하이브리드 성능 대시보드 (사이드바)
if show_performance and hasattr(st.session_state, 'hybrid_tracker'):
    with st.sidebar:
        st.header("📈 하이브리드 성능 대시보드")
        
        # Langsmith 상태
        langsmith_status = st.session_state.hybrid_tracker.get_langsmith_status()
        if langsmith_status['enabled']:
            st.success(f"📊 Langsmith: ✅ ({langsmith_status['project']})")
        else:
            st.warning("📊 Langsmith: ❌")
        st.success("🔧 커스텀 추적: ✅")
        
        # 하이브리드 인사이트
        hybrid_insights = st.session_state.hybrid_tracker.get_hybrid_insights()
        perf_summary = hybrid_insights['system_performance']
        
        if perf_summary:
            st.subheader("전체 작업 요약")
            total_time = sum(metrics['실행시간 (초)'] for metrics in perf_summary.values())
            total_python_memory = sum(metrics['Python 메모리 (MB)'] for metrics in perf_summary.values())
            total_ollama_memory = sum(metrics['Ollama 메모리 (MB)'] for metrics in perf_summary.values())
            total_memory = sum(metrics['총 메모리 (MB)'] for metrics in perf_summary.values())
            
            st.metric("총 처리 시간", f"{total_time:.2f}초")
            st.metric("Python 메모리 총합", f"{total_python_memory:.2f}MB")
            st.metric("Ollama 메모리 총합", f"{total_ollama_memory:.2f}MB")
            st.metric("전체 메모리 총합", f"{total_memory:.2f}MB")
            
            # 가장 오래 걸린 작업
            if perf_summary:
                slowest_task = max(perf_summary.items(), key=lambda x: x[1]['실행시간 (초)'])
                st.metric("가장 느린 작업", f"{slowest_task[0]}: {slowest_task[1]['실행시간 (초)']}초")
        
        # 현재 시스템 정보
        st.subheader("실시간 시스템 정보")
        memory_info = st.session_state.hybrid_tracker.system_tracker.get_total_memory_usage()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # 0.1초로 단축
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python 프로세스", f"{memory_info['python_memory']:.1f}MB")
            st.metric("Ollama 프로세스", f"{memory_info['ollama_memory']:.1f}MB")
        with col2:
            st.metric("총 메모리 사용", f"{memory_info['total_memory']:.1f}MB")
            st.metric("시스템 CPU 사용률", f"{cpu_percent:.1f}%")
            
        # Ollama 프로세스 정보
        if memory_info['ollama_process_count'] > 0:
            st.info(f"🤖 활성 Ollama 프로세스: {memory_info['ollama_process_count']}개")
        
        # 성능 추천사항
        recommendations = hybrid_insights['recommendations']
        if recommendations:
            st.subheader("🎯 최적화 추천")
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Langsmith 링크 (활성화된 경우)
        if langsmith_status['enabled']:
            st.subheader("🔗 Langsmith 대시보드")
            st.markdown(f"[Langsmith 웹 대시보드 열기](https://smith.langchain.com/)")
            st.caption(f"프로젝트: {langsmith_status['project']}")

if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.qa_chain is not None:
            with st.spinner("하이브리드 추적으로 LLM 응답을 생성 중입니다... (Langsmith + 시스템 모니터링)"):
                # 하이브리드 질문 응답 추적
                if show_performance:
                    # 문서 메타데이터 준비
                    metadata = {
                        'document_pages': st.session_state.processing_stats['문서_페이지_수'],
                        'document_chunks': st.session_state.processing_stats['청크_수'],
                        'query_length': len(prompt)
                    }
                    
                    # 하이브리드 추적으로 LLM 추론 실행
                    combined_result = st.session_state.hybrid_tracker.track_llm_inference(
                        st.session_state.qa_chain, 
                        prompt, 
                        metadata
                    )
                    
                    response = combined_result['response']
                    system_metrics = combined_result['system_metrics']
                    langsmith_enabled = combined_result['langsmith_enabled']
                else:
                    # 성능 측정 없이 실행
                    response = st.session_state.qa_chain({"query": prompt})
                    system_metrics = None
                    langsmith_enabled = False
                
                st.markdown(response["result"])
                
                # 하이브리드 성능 정보 표시
                if show_performance and system_metrics:
                    with st.expander("⚡ 하이브리드 응답 성능 정보"):
                        # 추적 시스템 상태
                        st.write("**추적 시스템:**")
                        track_col1, track_col2 = st.columns(2)
                        with track_col1:
                            st.success("🔧 커스텀 시스템 추적: ✅")
                        with track_col2:
                            if langsmith_enabled:
                                st.success("📊 Langsmith LLM 추적: ✅")
                            else:
                                st.warning("📊 Langsmith LLM 추적: ❌")
                        
                        # 시스템 성능 메트릭
                        st.write("**시스템 리소스:**")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("응답 시간", f"{system_metrics['duration']:.2f}초")
                            st.metric("완료 시간", datetime.fromtimestamp(system_metrics['end_time']).strftime('%H:%M:%S'))
                        with perf_col2:
                            st.metric("Python 메모리", f"{system_metrics['python_memory_used']:.2f}MB")
                            st.metric("Ollama 메모리", f"{system_metrics['ollama_memory_used']:.2f}MB")
                        with perf_col3:
                            st.metric("총 메모리 사용", f"{system_metrics['total_memory_used']:.2f}MB")
                            # 대략적인 처리 속도 계산
                            if system_metrics['duration'] > 0:
                                chars_per_sec = len(prompt) / system_metrics['duration']
                                st.metric("처리 속도", f"{chars_per_sec:.1f} 문자/초")
                        
                        # Langsmith 정보 (활성화된 경우)
                        if langsmith_enabled:
                            st.info("📊 **Langsmith에서 확인 가능한 정보:**\n- 프롬프트/응답 로그\n- 토큰 사용량\n- 체인 실행 흐름\n- 응답 품질 평가")
                
                with st.expander("📄 참고 문서 보기"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"**문서 {i+1}:**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.write("---")
        else:
            st.warning("먼저 PDF 파일을 처리해주세요.")
    
    if st.session_state.qa_chain is not None:
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
