"""
네비게이션 바 UI 컴포넌트
"""
import os
import streamlit as st
from core.performance import setup_langsmith


def render_navigation_bar():
    """상단 네비게이션 바 렌더링"""
    col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 2])
    
    with col_nav3:
        # Langsmith 상태 실시간 확인
        _, langsmith_enabled = setup_langsmith()
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "unified-rag-system")
        
        if langsmith_enabled:
            # LangSmith 관리 섹션
            with st.expander("📊 LangSmith 관리(소유자만)"):
                langsmith_url = "https://smith.langchain.com"
                
                st.markdown(f"""
                **프로젝트:** `{langsmith_project}`  
                **상태:** 🟢 활성화됨
                """)
                
                if st.button("📈 대시보드", key="langsmith_dashboard"):
                    st.markdown(f'''
                    <script>
                    window.open("{langsmith_url}", "_blank");
                    </script>
                    ''', unsafe_allow_html=True)
                    st.info(f"LangSmith 대시보드가 새 탭에서 열립니다: {langsmith_url}")
                
                st.markdown("**주요 기능:**")
                st.write("• 실시간 추론 추적")
                st.write("• 성능 메트릭 분석")
                st.write("• 오류 디버깅")
                st.write("• 비용 모니터링")
                st.markdown("**확인 요소:**")
                st.write("• 프롬프트/응답 로그")
                st.write("• 토큰 사용량")
                st.write("• 체인 실행 흐름")
                st.write("• 응답 품질 평가")
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
                    st.markdown('''
                    <script>
                    window.open("https://smith.langchain.com", "_blank");
                    </script>
                    ''', unsafe_allow_html=True)
                    st.info("LangSmith 웹사이트가 새 탭에서 열립니다: https://smith.langchain.com")
    
    with col_nav2:
        # 추가 네비게이션 공간 (필요시 확장 가능)
        st.empty()
    
    st.divider()
