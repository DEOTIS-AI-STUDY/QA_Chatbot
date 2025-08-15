"""
통합 RAG 시스템 - 리팩토링된 메인 애플리케이션
"""
import streamlit as st
from ui.common import initialize_session_state
from ui.navigation import render_navigation_bar
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface
from ui.performance import render_performance_monitor


def main():
    """메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="통합 RAG 시스템",
        page_icon="🤖",
        layout="wide"
    )
    
    # 제목
    st.title("🚀 통합 RAG 시스템")
    st.markdown("**BGE-M3 임베딩 + Elasticsearch + 멀티 LLM + 성능 모니터링**")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 네비게이션 바 렌더링
    render_navigation_bar()
    
    # 사이드바 렌더링
    model_choice, top_k, available_models = render_sidebar()
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 채팅 인터페이스
        render_chat_interface(available_models, model_choice, top_k)
    
    with col2:
        # 성능 모니터링
        render_performance_monitor()


if __name__ == "__main__":
    main()
