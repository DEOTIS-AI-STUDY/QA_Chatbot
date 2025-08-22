"""
Streamlit UI 공통 유틸리티
"""
import streamlit as st
import os
import uuid
from core.performance import HybridPerformanceTracker
from core.chat_history import ChatHistoryManager, StreamlitChatHistoryInterface
from core.models import ModelFactory
from utils.elasticsearch import ElasticsearchManager


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
    # 공통 대화 기록 관리자 초기화
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatHistoryManager(max_history=15)
    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = StreamlitChatHistoryInterface(st.session_state.chat_manager)
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "auto_indexing_done" not in st.session_state:
        st.session_state.auto_indexing_done = False


def reset_rag_state():
    """RAG 상태 리셋"""
    st.session_state.qa_chain = None
    st.session_state.rag_initialized = False
    st.session_state.selected_model = None
    st.session_state.messages = []
    # 공통 대화 기록 관리자 리셋
    if hasattr(st.session_state, 'chat_manager'):
        st.session_state.chat_manager.clear_history()
    st.success("상태가 리셋되었습니다. 다시 초기화해주세요.")


def show_debug_info():
    """디버그 정보 표시"""
    with st.expander("🔧 디버그 정보"):
        st.write(f"qa_chain: {st.session_state.qa_chain}")
        st.write(f"qa_chain type: {type(st.session_state.qa_chain)}")
        st.write(f"qa_chain is None: {st.session_state.qa_chain is None}")


# 공통 모듈 사용으로 대체된 함수들
def get_chat_manager():
    """공통 대화 기록 관리자 반환"""
    return st.session_state.chat_manager


def get_chat_interface():
    """공통 대화 기록 인터페이스 반환"""
    return st.session_state.chat_interface


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
