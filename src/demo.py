import streamlit as st
import os
import time
import psutil
from datetime import datetime
from tempfile import NamedTemporaryFile

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

# 성능 측정 유틸리티 함수들
class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
        
    def start_timer(self, task_name):
        """작업 시작 시간 기록"""
        self.metrics[task_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        
    def end_timer(self, task_name):
        """작업 종료 시간 기록 및 결과 반환"""
        if task_name in self.metrics:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.metrics[task_name].update({
                'end_time': end_time,
                'end_memory': end_memory,
                'duration': end_time - self.metrics[task_name]['start_time'],
                'memory_used': end_memory - self.metrics[task_name]['start_memory']
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
                    '메모리 사용량 (MB)': round(metrics['memory_used'], 2),
                    '완료시간': datetime.fromtimestamp(metrics['end_time']).strftime('%H:%M:%S')
                }
        return summary

# 전역 성능 추적기 초기화
if "performance_tracker" not in st.session_state:
    st.session_state.performance_tracker = PerformanceTracker()

# 1. 핵심 기능 함수 정의
def process_pdf_and_create_qa_chain(pdf_file, tracker):
    """PDF를 처리하여 QA 체인을 생성하는 함수"""
    
    # 전체 처리 시간 추적 시작
    tracker.start_timer("전체_PDF_처리")
    
    # 1. PDF 로딩
    tracker.start_timer("PDF_로딩")
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    tracker.end_timer("PDF_로딩")
    
    # 문서 정보 저장
    doc_count = len(documents)
    total_chars = sum(len(doc.page_content) for doc in documents)

    # 2. 텍스트 분할
    tracker.start_timer("텍스트_분할")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    chunk_count = len(docs)
    tracker.end_timer("텍스트_분할")

    # 3. 임베딩 모델 로딩
    tracker.start_timer("임베딩_모델_로딩")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} # CPU 사용, GPU 사용 시 'cuda'로 변경
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    tracker.end_timer("임베딩_모델_로딩")
    
    # 4. 벡터 스토어 생성 (임베딩 생성)
    tracker.start_timer("벡터_스토어_생성")
    vectorstore = FAISS.from_documents(docs, embeddings)
    tracker.end_timer("벡터_스토어_생성")

    # 5. QA 체인 구성
    tracker.start_timer("QA_체인_구성")
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
    tracker.end_timer("QA_체인_구성")
    
    # 전체 처리 완료
    tracker.end_timer("전체_PDF_처리")
    
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
st.set_page_config(page_title="로컬 PDF QA 챗봇", page_icon="🤖")
st.title("📄 로컬 PDF 기반 QA 챗봇 (Ollama)")

# 성능 모니터링 토글
show_performance = st.sidebar.checkbox("🔍 성능 모니터링 활성화", value=True)

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
            with st.spinner("로컬 모델로 PDF를 처리 중입니다... (처음에는 모델 다운로드로 오래 걸릴 수 있습니다)"):
                # 성능 추적기 초기화
                st.session_state.performance_tracker = PerformanceTracker()
                
                # PDF 처리 및 QA 체인 생성
                qa_chain, processing_stats = process_pdf_and_create_qa_chain(
                    uploaded_file, 
                    st.session_state.performance_tracker
                )
                
                st.session_state.qa_chain = qa_chain
                st.session_state.processing_stats = processing_stats
                st.session_state.messages = [{"role": "assistant", "content": "PDF 처리가 완료되었습니다. 무엇이든 물어보세요!"}]
            
            st.success("PDF 처리가 완료되었습니다!")
            
            # 성능 결과 표시
            if show_performance:
                st.subheader("📊 PDF 처리 성능 분석")
                
                # 처리 통계
                st.write("**문서 정보:**")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("페이지 수", f"{processing_stats['문서_페이지_수']}개")
                    st.metric("청크 수", f"{processing_stats['청크_수']}개")
                with stats_col2:
                    st.metric("총 문자 수", f"{processing_stats['총_문자_수']:,}자")
                    st.metric("검색 문서 수", f"{processing_stats['검색_문서_수']}개")
                
                # 성능 메트릭
                perf_summary = st.session_state.performance_tracker.get_summary()
                st.write("**처리 시간 분석:**")
                
                for task, metrics in perf_summary.items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{task}", f"{metrics['실행시간 (초)']}초")
                    with col2:
                        st.metric("메모리 사용", f"{metrics['메모리 사용량 (MB)']}MB")
                    with col3:
                        st.metric("완료 시간", metrics['완료시간'])
                
        else:
            st.error("PDF 파일을 업로드해주세요.")

# 채팅 기록 표시 및 사용자 입력 처리 (동일)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 성능 대시보드 (사이드바)
if show_performance and hasattr(st.session_state, 'performance_tracker'):
    with st.sidebar:
        st.header("📈 성능 대시보드")
        
        # 전체 성능 요약
        perf_summary = st.session_state.performance_tracker.get_summary()
        if perf_summary:
            st.subheader("전체 작업 요약")
            total_time = sum(metrics['실행시간 (초)'] for metrics in perf_summary.values())
            total_memory = sum(metrics['메모리 사용량 (MB)'] for metrics in perf_summary.values())
            
            st.metric("총 처리 시간", f"{total_time:.2f}초")
            st.metric("총 메모리 사용", f"{total_memory:.2f}MB")
            
            # 가장 오래 걸린 작업
            if perf_summary:
                slowest_task = max(perf_summary.items(), key=lambda x: x[1]['실행시간 (초)'])
                st.metric("가장 느린 작업", f"{slowest_task[0]}: {slowest_task[1]['실행시간 (초)']}초")
        
        # 현재 시스템 정보
        st.subheader("시스템 정보")
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=1)
        
        st.metric("현재 메모리", f"{current_memory:.1f}MB")
        st.metric("CPU 사용률", f"{cpu_percent:.1f}%")

if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.qa_chain is not None:
            with st.spinner("로컬 LLM이 답변을 생성 중입니다..."):
                # 질문 응답 성능 측정
                if show_performance:
                    st.session_state.performance_tracker.start_timer("질문_응답")
                
                response = st.session_state.qa_chain({"query": prompt})
                
                if show_performance:
                    qa_metrics = st.session_state.performance_tracker.end_timer("질문_응답")
                
                st.markdown(response["result"])
                
                # 성능 정보 표시
                if show_performance and qa_metrics:
                    with st.expander("⚡ 응답 성능 정보"):
                        perf_col1, perf_col2 = st.columns(2)
                        with perf_col1:
                            st.metric("응답 시간", f"{qa_metrics['duration']:.2f}초")
                            st.metric("완료 시간", datetime.fromtimestamp(qa_metrics['end_time']).strftime('%H:%M:%S'))
                        with perf_col2:
                            st.metric("메모리 사용", f"{qa_metrics['memory_used']:.2f}MB")
                            # 대략적인 처리 속도 계산
                            if qa_metrics['duration'] > 0:
                                chars_per_sec = len(prompt) / qa_metrics['duration']
                                st.metric("처리 속도", f"{chars_per_sec:.1f} 문자/초")
                
                with st.expander("📄 참고 문서 보기"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"**문서 {i+1}:**")
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.write("---")
        else:
            st.warning("먼저 PDF 파일을 처리해주세요.")
    
    if st.session_state.qa_chain is not None:
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
