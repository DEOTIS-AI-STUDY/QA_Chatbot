"""
채팅 인터페이스 UI 컴포넌트
"""
import streamlit as st
from core.models import ModelFactory
from core.rag import create_rag_chain
from core.config import LLM_MODELS
from ui.diagnostics import show_initialization_diagnostics


def render_chat_interface(available_models, model_choice, top_k):
    """채팅 인터페이스 렌더링"""
    st.subheader("💬 대화")
    
    # RAG 시스템 상태 표시 및 초기화 버튼
    warning_container = _show_rag_status()
    _render_initialization_button(available_models, model_choice, top_k, warning_container)
    
    # 채팅 메시지 표시
    _display_chat_messages()
    
    # 사용자 입력 처리
    _handle_user_input(top_k)


def _show_rag_status():
    """RAG 시스템 상태 표시"""
    warning_container = st.empty()
    
    if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
        st.success(f"✅ RAG 시스템이 초기화되어 있습니다. (모델: {st.session_state.selected_model})")
    elif st.session_state.rag_initialized and st.session_state.qa_chain is None:
        st.error("❌ RAG 시스템 초기화에 오류가 있습니다. 다시 초기화해주세요.")
        st.info("상태가 일치하지 않습니다. 아래 '상태 리셋' 버튼을 사용하세요.")
    elif not st.session_state.rag_initialized:
        warning_container.warning("⚠️ RAG 시스템이 초기화되지 않았습니다.")
        if st.session_state.qa_chain is False:
            st.error("❌ 이전 초기화에서 오류가 발생했습니다. 다시 초기화해주세요.")
    
    return warning_container


def _render_initialization_button(available_models, model_choice, top_k, warning_container):
    """RAG 시스템 초기화 버튼 렌더링"""
    if st.button("🔧 RAG 시스템 초기화"):
        if not available_models:
            st.error("❌ 사용 가능한 LLM 모델이 없습니다.")
            return
        
        _initialize_rag_system(model_choice, top_k, warning_container)


def _initialize_rag_system(model_choice, top_k, warning_container):
    """RAG 시스템 초기화"""
    with st.spinner("RAG 시스템 초기화 중..."):
        try:
            current_model = st.session_state.get('selected_model', model_choice)
            
            # 임베딩 모델 생성
            st.write("📝 임베딩 모델 생성 중...")
            embeddings = ModelFactory.create_embedding_model()
            if embeddings is None:
                st.error("❌ 임베딩 모델 생성 실패")
                return
            st.write("✅ 임베딩 모델 생성 완료")
            
            # LLM 모델 생성
            st.write("🤖 LLM 모델 생성 중...")
            llm_model, message = ModelFactory.create_llm_model(current_model)
            if llm_model is None:
                st.error(f"❌ LLM 모델 생성 실패: {message}")
                return
            st.write(f"✅ {message}")
            
            # RAG 체인 생성
            st.write("🔗 RAG 체인 생성 중...")
            qa_chain, success_or_error = create_rag_chain(embeddings, llm_model, top_k)
            
            _handle_initialization_result(qa_chain, success_or_error, current_model, warning_container)
            
        except Exception as e:
            st.session_state.qa_chain = None
            st.session_state.rag_initialized = False
            st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
            show_initialization_diagnostics(e, model_choice)


def _handle_initialization_result(qa_chain, success_or_error, current_model, warning_container):
    """초기화 결과 처리"""
    if success_or_error is True and qa_chain is not None:
        st.session_state.qa_chain = qa_chain
        st.session_state.selected_model = current_model
        st.session_state.rag_initialized = True
        
        # 경고 메시지 즉시 제거
        warning_container.empty()
        
        # 초기화 메시지 추가
        if not st.session_state.messages:
            st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"RAG 시스템이 초기화되었습니다. 모델: {LLM_MODELS[current_model]['name']}"
        })
        
        st.success("🎉 RAG 시스템 초기화 완료!")
        st.info("이제 아래에서 질문하실 수 있습니다.")
    else:
        st.session_state.qa_chain = None
        st.session_state.rag_initialized = False
        
        # 오류 메시지 처리
        if success_or_error is True:
            error_msg = "qa_chain이 None으로 반환됨 (내부 로직 오류)"
        elif isinstance(success_or_error, str):
            error_msg = success_or_error
        else:
            error_msg = f"알 수 없는 반환값: {success_or_error}"
            
        st.error(f"❌ RAG 시스템 초기화 실패: {error_msg}")
        show_initialization_diagnostics(None, st.session_state.get('selected_model'))


def _display_chat_messages():
    """채팅 메시지 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_input(top_k):
    """사용자 입력 처리"""
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.qa_chain is not None and st.session_state.rag_initialized:
                _process_query(prompt, top_k)
            else:
                st.warning("먼저 RAG 시스템을 초기화해주세요.")
                st.info("위의 '🔧 RAG 시스템 초기화' 버튼을 클릭하세요.")
                if st.session_state.rag_initialized and st.session_state.qa_chain is None:
                    st.error("⚠️ 시스템 상태 불일치가 감지되었습니다. 사이드바의 '상태 리셋' 버튼을 사용하세요.")


def _process_query(prompt, top_k):
    """쿼리 처리"""
    with st.spinner("답변 생성 중..."):
        current_model = st.session_state.get('selected_model')
        
        # 하이브리드 추적으로 LLM 추론
        metadata = {
            'model': LLM_MODELS[current_model]['name'] if current_model else 'Unknown',
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
        
        # 답변 표시
        st.markdown(response["result"])
        
        # 성능 정보 표시
        _show_performance_info(system_metrics)
        
        # 소스 문서 표시
        _show_source_documents(response["source_documents"])
        
        # 메시지 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})


def _show_performance_info(system_metrics):
    """성능 정보 표시"""
    if system_metrics:
        with st.expander("⚡ 성능 정보"):
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("응답 시간", f"{system_metrics['duration']:.2f}초")
            with perf_col2:
                st.metric("총 메모리", f"{system_metrics['total_memory_used']:.2f}MB")
            with perf_col3:
                st.metric("ES 프로세스", system_metrics['elasticsearch_process_count'])


def _show_source_documents(source_documents):
    """소스 문서 표시"""
    with st.expander("📄 참고 문서"):
        for i, doc in enumerate(source_documents):
            st.write(f"**문서 {i+1}:**")
            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            st.write(f"*출처: {doc.metadata.get('filename', 'Unknown')}*")
            st.divider()
