"""
Streamlit UI 공통 유틸리티
"""
import streamlit as st
import uuid
from core.performance import HybridPerformanceTracker


def initialize_session_state():
    """세션 상태 초기화"""
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


def reset_rag_state():
    """RAG 상태 리셋"""
    st.session_state.qa_chain = None
    st.session_state.rag_initialized = False
    st.session_state.selected_model = None
    st.success("상태가 리셋되었습니다. 다시 초기화해주세요.")


def show_debug_info():
    """디버그 정보 표시"""
    with st.expander("🔧 디버그 정보"):
        st.write(f"qa_chain: {st.session_state.qa_chain}")
        st.write(f"qa_chain type: {type(st.session_state.qa_chain)}")
        st.write(f"qa_chain is None: {st.session_state.qa_chain is None}")
        st.write(f"qa_chain == False: {st.session_state.qa_chain == False}")
        st.write(f"bool(qa_chain): {bool(st.session_state.qa_chain)}")
        st.write(f"rag_initialized: {st.session_state.get('rag_initialized', False)}")
        st.write(f"selected_model: {st.session_state.get('selected_model', 'None')}")
        
        # 상태 리셋 버튼
        if st.button("🔄 상태 리셋", key="reset_debug"):
            reset_rag_state()


def show_rag_status():
    """RAG 시스템 상태 표시"""
    warning_container = st.empty()
    
    if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
        st.success(f"✅ RAG 시스템이 초기화되어 있습니다. (모델: {st.session_state.selected_model})")
        return warning_container, True
    elif st.session_state.rag_initialized and st.session_state.qa_chain is None:
        st.error("❌ RAG 시스템 초기화에 오류가 있습니다. 다시 초기화해주세요.")
        st.info("상태가 일치하지 않습니다. 아래 '상태 리셋' 버튼을 사용하세요.")
        return warning_container, False
    elif not st.session_state.rag_initialized:
        # RAG 시스템이 초기화되지 않은 경우에만 경고 메시지 표시
        warning_container.warning("⚠️ RAG 시스템이 초기화되지 않았습니다.")
        if st.session_state.qa_chain is False:
            st.error("❌ 이전 초기화에서 오류가 발생했습니다. 다시 초기화해주세요.")
        return warning_container, False
    
    return warning_container, False
