"""
네비게이션 바 UI 컴포넌트
"""
import os
import streamlit as st


def render_navigation_bar():
    """상단 네비게이션 바 렌더링"""
    col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 2])
    
    with col_nav3:
        # LangSmith 비활성화됨
        langsmith_enabled = False
        
        if langsmith_enabled:
            # LangSmith는 비활성화되었으므로 이 블록은 실행되지 않음
            pass
        else:
            st.info("📊 LangSmith: 비활성화됨 (Rate Limit 초과로 제거)")
    
    with col_nav2:
        # 추가 네비게이션 공간 (필요시 확장 가능)
        st.empty()
    
    st.divider()
