"""
사이드바 UI 컴포넌트
"""
import os
import streamlit as st
from core.models import ModelFactory
from core.config import LLM_MODELS, PDF_DIR
from utils.elasticsearch import ElasticsearchManager
from ui.common import show_debug_info, reset_rag_state


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        # LLM 모델 선택
        st.subheader("🤖 LLM 모델 선택")
        
        # 사용 가능한 모델 필터링
        available_models = ModelFactory.get_available_models()
            
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
            _show_model_status(model_choice)
        
        # 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("검색할 문서 수", 1, 10, 3)
        
        # 디버깅 정보
        show_debug_info()
        
        st.divider()
        
        # PDF 관리
        _render_pdf_management()

        st.divider()

        st.header("⚙️ 시스템 설정")
        
        # Elasticsearch 연결 상태
        _show_elasticsearch_status()
        
    return model_choice, top_k, available_models


def _show_model_status(model_choice):
    """모델 상태 표시"""
    if model_choice == "upstage":
        api_key = os.getenv("UPSTAGE_API_KEY")
        if api_key and api_key != "your_upstage_api_key":
            st.success("✅ Upstage API 키 설정됨")
        else:
            st.warning("⚠️ UPSTAGE_API_KEY 환경변수가 필요합니다")
    elif model_choice in ["qwen2", "llama3", "solar_10_7b"]:
        if model_choice == "solar_10_7b":
            st.info("ℹ️ SOLAR-10.7B 오픈소스 모델 - Ollama 서버가 실행 중인지 확인하세요")
        else:
            st.info("ℹ️ Ollama 서버가 실행 중인지 확인하세요")


def _render_pdf_management():
    """PDF 관리 섹션 렌더링"""
    st.subheader("📄 PDF 관리")
    
    # PDF 업로드
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📥 파일 업로드 및 인덱싱"):
        _handle_pdf_upload(uploaded_files)
    
    # 기존 PDF 파일 목록
    _show_existing_pdfs()


def _handle_pdf_upload(uploaded_files):
    """PDF 업로드 처리"""
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
        from core.models import ModelFactory
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


def _show_existing_pdfs():
    """기존 PDF 파일 목록 표시"""
    existing_pdfs = ElasticsearchManager.list_pdfs(PDF_DIR)
    if existing_pdfs:
        st.write("**기존 PDF 파일:**")
        for pdf in existing_pdfs:
            st.write(f"• {os.path.basename(pdf)}")
        
        if st.button("🔄 기존 파일 재인덱싱"):
            with st.spinner("기존 PDF 파일 재인덱싱 중..."):
                from core.models import ModelFactory
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


def _show_elasticsearch_status():
    """Elasticsearch 연결 상태 표시"""
    es_connected, es_message = ElasticsearchManager.check_connection()
    if es_connected:
        st.success(f"✅ Elasticsearch: {es_message}")
    else:
        st.error(f"❌ Elasticsearch: {es_message}")
        st.stop()
