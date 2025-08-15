"""
성능 모니터링 UI 컴포넌트
"""
import psutil
import streamlit as st


def render_performance_monitor():
    """성능 모니터링 패널 렌더링"""
    with st.expander("📊 성능 모니터링", expanded=True):
        st.subheader("📈 성능 대시보드")
        
        # 현재 시스템 정보
        _show_current_metrics()
        
        # 하이브리드 인사이트
        _show_hybrid_insights()


def _show_current_metrics():
    """현재 시스템 메트릭 표시"""
    memory_info = st.session_state.hybrid_tracker.system_tracker.get_total_memory_usage()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    st.metric("시스템 CPU", f"{cpu_percent:.1f}%")
    st.metric("Python 메모리", f"{memory_info['python_memory']:.1f}MB")
    st.metric("Elasticsearch", f"{memory_info['elasticsearch_memory']:.1f}MB")
    st.metric("Ollama", f"{memory_info['ollama_memory']:.1f}MB")
    st.metric("총 메모리", f"{memory_info['total_memory']:.1f}MB")


def _show_hybrid_insights():
    """하이브리드 인사이트 표시"""
    hybrid_insights = st.session_state.hybrid_tracker.get_hybrid_insights()
    
    # 성능 요약
    perf_summary = hybrid_insights['system_performance']
    if perf_summary:
        st.subheader("작업 요약")
        for task, metrics in perf_summary.items():
            st.write(f"**{task}**: {metrics['실행시간 (초)']}초")
    
    # 추천사항
    recommendations = hybrid_insights['recommendations']
    if recommendations:
        st.subheader("🎯 최적화 추천")
        for rec in recommendations:
            st.write(f"• {rec}")
