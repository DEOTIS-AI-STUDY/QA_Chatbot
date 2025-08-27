"""
성능 추적 및 모니터링 모듈
"""
import os
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any


class PerformanceTracker:
    """시스템 성능 추적 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.current_process = psutil.Process()
        
    def get_elasticsearch_memory(self) -> tuple[float, int]:
        """Elasticsearch 프로세스 메모리 사용량"""
        es_memory = 0
        es_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'elasticsearch' in proc.info['name'].lower() or 'java' in proc.info['name'].lower():
                    # Java 프로세스 중 Elasticsearch 관련 확인
                    try:
                        cmdline = proc.cmdline()
                        if any('elasticsearch' in cmd.lower() for cmd in cmdline):
                            es_processes.append(proc)
                            es_memory += proc.memory_info().rss / 1024 / 1024
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return es_memory, len(es_processes)
    
    def get_ollama_memory(self) -> tuple[float, int]:
        """Ollama 프로세스 메모리 사용량"""
        ollama_memory = 0
        ollama_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
                    ollama_memory += proc.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return ollama_memory, len(ollama_processes)
    
    def get_total_memory_usage(self) -> Dict[str, float]:
        """전체 시스템 메모리 사용량"""
        python_memory = self.current_process.memory_info().rss / 1024 / 1024
        es_memory, es_count = self.get_elasticsearch_memory()
        ollama_memory, ollama_count = self.get_ollama_memory()
        
        return {
            'python_memory': python_memory,
            'elasticsearch_memory': es_memory,
            'ollama_memory': ollama_memory,
            'total_memory': python_memory + es_memory + ollama_memory,
            'elasticsearch_process_count': es_count,
            'ollama_process_count': ollama_count
        }
        
    def start_timer(self, task_name: str) -> None:
        """작업 시작 시간 기록"""
        memory_info = self.get_total_memory_usage()
        self.metrics[task_name] = {
            'start_time': time.time(),
            'start_python_memory': memory_info['python_memory'],
            'start_elasticsearch_memory': memory_info['elasticsearch_memory'],
            'start_ollama_memory': memory_info['ollama_memory'],
            'start_total_memory': memory_info['total_memory'],
            'start_cpu_percent': self.current_process.cpu_percent()
        }
        
    def end_timer(self, task_name: str) -> Optional[Dict[str, Any]]:
        """작업 종료 시간 기록 및 결과 반환"""
        if task_name in self.metrics:
            end_time = time.time()
            memory_info = self.get_total_memory_usage()
            
            self.metrics[task_name].update({
                'end_time': end_time,
                'end_python_memory': memory_info['python_memory'],
                'end_elasticsearch_memory': memory_info['elasticsearch_memory'],
                'end_ollama_memory': memory_info['ollama_memory'],
                'end_total_memory': memory_info['total_memory'],
                'duration': end_time - self.metrics[task_name]['start_time'],
                'python_memory_used': memory_info['python_memory'] - self.metrics[task_name]['start_python_memory'],
                'elasticsearch_memory_used': memory_info['elasticsearch_memory'] - self.metrics[task_name]['start_elasticsearch_memory'],
                'ollama_memory_used': memory_info['ollama_memory'] - self.metrics[task_name]['start_ollama_memory'],
                'total_memory_used': memory_info['total_memory'] - self.metrics[task_name]['start_total_memory'],
                'elasticsearch_process_count': memory_info['elasticsearch_process_count'],
                'ollama_process_count': memory_info['ollama_process_count']
            })
            
            return self.metrics[task_name]
        return None
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """전체 성능 요약 반환"""
        summary = {}
        for task, metrics in self.metrics.items():
            if 'duration' in metrics:
                summary[task] = {
                    '실행시간 (초)': round(metrics['duration'], 2),
                    'Python 메모리 (MB)': round(metrics['python_memory_used'], 2),
                    'Elasticsearch 메모리 (MB)': round(metrics['elasticsearch_memory_used'], 2),
                    'Ollama 메모리 (MB)': round(metrics['ollama_memory_used'], 2),
                    '총 메모리 (MB)': round(metrics['total_memory_used'], 2),
                    'ES 프로세스 수': metrics['elasticsearch_process_count'],
                    'Ollama 프로세스 수': metrics['ollama_process_count'],
                    '완료시간': datetime.fromtimestamp(metrics['end_time']).strftime('%H:%M:%S')
                }
        return summary


class HybridPerformanceTracker:
    """하이브리드 성능 추적 클래스"""
    
    def __init__(self):
        self.system_tracker = PerformanceTracker()
        self.hybrid_metrics = {
            'system_metrics': {},
            'combined_insights': {}
        }
        
    def get_langsmith_status(self) -> Dict[str, Any]:
        """Langsmith 상태 반환 (비활성화됨)"""
        return {
            'available': False,
            'enabled': False,
            'project': 'disabled'
        }
    
    def track_preprocessing_stage(self, stage_name: str) -> None:
        """전처리 단계 추적"""
        return self.system_tracker.start_timer(stage_name)
    
    def end_preprocessing_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """전처리 단계 종료"""
        metrics = self.system_tracker.end_timer(stage_name)
        if metrics:
            self.hybrid_metrics['system_metrics'][stage_name] = metrics
        return metrics
    
    def track_llm_inference(self, qa_chain, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """LLM 추론 하이브리드 추적"""
        import streamlit as st
        
        self.system_tracker.start_timer("LLM_추론_시스템")
        
        if metadata is None:
            metadata = {}
        
        enhanced_metadata = {
            **metadata,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'system': 'unified_rag'
        }
        
        try:
            response = qa_chain({"query": query})
            
            system_metrics = self.system_tracker.end_timer("LLM_추론_시스템")
            
            combined_result = {
                'response': response,
                'system_metrics': system_metrics,
                'langsmith_enabled': False,
                'metadata': enhanced_metadata
            }
            
            return combined_result
            
        except Exception as e:
            self.system_tracker.end_timer("LLM_추론_시스템")
            raise
    
    def get_system_summary(self) -> Dict[str, Dict[str, Any]]:
        """시스템 성능 요약"""
        return self.system_tracker.get_summary()
    
    def get_hybrid_insights(self) -> Dict[str, Any]:
        """하이브리드 성능 인사이트"""
        system_summary = self.get_system_summary()
        
        insights = {
            'system_performance': system_summary,
            'langsmith_status': self.get_langsmith_status(),
            'recommendations': self._generate_recommendations(system_summary)
        }
        
        return insights
    
    def _generate_recommendations(self, system_summary: Dict[str, Dict[str, Any]]) -> List[str]:
        """성능 기반 추천사항 생성"""
        recommendations = []
        
        if system_summary:
            slowest_task = max(system_summary.items(), 
                             key=lambda x: x[1]['실행시간 (초)'], 
                             default=(None, {'실행시간 (초)': 0}))
            
            if slowest_task[0] and slowest_task[1]['실행시간 (초)'] > 10:
                recommendations.append(f"⚠️ {slowest_task[0]}이 {slowest_task[1]['실행시간 (초)']}초로 가장 오래 걸립니다.")
                
                if 'Elasticsearch' in slowest_task[0]:
                    recommendations.append("💡 Elasticsearch 인덱스 설정을 최적화하거나 더 적은 문서로 테스트해보세요.")
                elif 'LLM_추론' in slowest_task[0]:
                    recommendations.append("💡 더 작은 LLM 모델을 선택하거나 GPU를 사용해보세요.")
            
            total_memory = sum(metrics.get('총 메모리 (MB)', 0) for metrics in system_summary.values())
            if total_memory > 8000:
                recommendations.append(f"🔥 총 메모리 사용량이 {total_memory:.1f}MB로 높습니다. 시스템 최적화를 고려해보세요.")
        
        return recommendations
