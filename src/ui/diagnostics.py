"""
시스템 진단 UI 컴포넌트
"""
import os
import requests
import streamlit as st
import traceback
from core.config import INDEX_NAME, LLM_MODELS
from core.models import ModelFactory
from utils.elasticsearch import ElasticsearchManager


def show_initialization_diagnostics(exception, current_model):
    """초기화 진단 시스템"""
    st.info("🔍 자동 진단을 실행합니다...")
    with st.expander("📋 상세 진단 결과", expanded=True):
        _diagnose_elasticsearch()
        st.divider()
        _diagnose_llm_model(current_model)
        st.divider()
        _diagnose_environment(current_model)
        st.divider()
        _show_recommendations()
        _show_quick_actions()
        
        if exception:
            _show_exception_details(exception)


def _diagnose_elasticsearch():
    """Elasticsearch 진단"""
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


def _diagnose_llm_model(current_model):
    """LLM 모델 진단"""
    st.write("**2. LLM 모델 가용성 확인**")
    try:
        test_llm, message = ModelFactory.create_llm_model(current_model)
        if test_llm is not None:
            # 간단한 테스트 호출
            try:
                if current_model == "upstage":
                    # Upstage API 테스트
                    test_response = test_llm.invoke("안녕하세요")
                    st.success(f"✅ {LLM_MODELS[current_model]['name']} 모델이 정상 작동합니다.")
                else:
                    # Ollama 모델 테스트 (간단한 ping)
                    _test_ollama_model(current_model)
            except Exception as test_e:
                st.warning(f"⚠️ 모델 생성은 됐지만 테스트 호출 실패: {str(test_e)}")
                if current_model in ["qwen2", "llama3"]:
                    st.info("💡 Ollama 서버가 실행 중이고 모델이 다운로드되어 있는지 확인하세요.")
        else:
            st.error(f"❌ {LLM_MODELS[current_model]['name']} 모델 생성 실패: {message}")
    except Exception as llm_e:
        st.error(f"❌ LLM 모델 진단 실패: {str(llm_e)}")


def _test_ollama_model(current_model):
    """Ollama 모델 테스트"""
    try:
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


def _diagnose_environment(current_model):
    """환경변수 및 API 키 진단"""
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
    from core.config import ELASTICSEARCH_URL, PDF_DIR
    env_vars = {
        "ELASTICSEARCH_URL": ELASTICSEARCH_URL,
        "INDEX_NAME": INDEX_NAME,
        "PDF_DIR": PDF_DIR
    }
    for var, value in env_vars.items():
        st.write(f"• {var}: `{value}`")


def _show_recommendations():
    """추천 해결 방법"""
    st.write("**4. 추천 해결 방법**")
    st.info("📝 **단계별 해결 방법:**")
    st.write("1️⃣ PDF 파일이 업로드되어 있는지 확인")
    st.write("2️⃣ Elasticsearch가 실행 중인지 확인 (Docker Compose 사용)")
    st.write("3️⃣ 선택한 LLM 모델 서비스가 실행 중인지 확인")
    st.write("4️⃣ 필요한 경우 API 키가 올바르게 설정되었는지 확인")
    st.write("5️⃣ '상태 리셋' 후 다시 초기화")


def _show_quick_actions():
    """빠른 액션 버튼들"""
    st.write("**빠른 액션:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 상태 리셋", key="quick_reset"):
            from ui.common import reset_rag_state
            reset_rag_state()
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
            current_model = st.session_state.get('selected_model')
            test_model, message = ModelFactory.create_llm_model(current_model)
            if test_model:
                st.success(f"{current_model} 모델 생성 성공!")
            else:
                st.error(f"{current_model} 모델 생성 실패: {message}")


def _show_exception_details(exception):
    """예외 상세 정보"""
    with st.expander("🔍 오류 진단 정보", expanded=True):
        st.write("**오류 상세 정보:**")
        st.code(str(exception))
        
        st.write("**가능한 원인:**")
        error_str = str(exception).lower()
        if "elasticsearch" in error_str:
            st.write("• Elasticsearch 연결 문제")
            st.info("💡 Elasticsearch가 실행 중인지 확인하세요: `docker-compose up elasticsearch`")
        elif "ollama" in error_str or "connection" in error_str:
            st.write("• Ollama 서버 연결 문제")
            st.info("💡 Ollama 서버를 시작하세요: `ollama serve`")
        elif "api" in error_str or "key" in error_str:
            st.write("• API 키 관련 문제")
            st.info("💡 .env 파일의 API 키를 확인하세요")
        else:
            st.write("• 알 수 없는 오류")
            st.info("💡 로그를 확인하고 필요시 시스템을 재시작하세요")
        
        if hasattr(exception, '__traceback__'):
            with st.expander("상세 트레이스백"):
                st.code(traceback.format_exc())
