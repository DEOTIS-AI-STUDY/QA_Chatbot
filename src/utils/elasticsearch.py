"""
Elasticsearch 검색 및 연결 관리 유틸리티
"""
import os
import urllib3
from typing import List, Tuple, Optional, Dict, Any
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import (
    ELASTICSEARCH_URL, 
    ELASTICSEARCH_HOST, 
    ELASTICSEARCH_PORT, 
    ELASTICSEARCH_SCHEME,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    INDEX_NAME
)


def get_optimized_text_splitter():
    """한국어 QA 최적화된 텍스트 분할기 반환
    
    Returns:
        RecursiveCharacterTextSplitter: 한국어에 최적화된 설정의 텍스트 분할기
        
    Features:
        - chunk_size: 800 (한국어는 더 작은 청크가 효과적)
        - chunk_overlap: 160 (20% 오버랩으로 문맥 연결성 향상)
        - separators: 한국어 문장 구분자 추가
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=800,  # 한국어는 더 작은 청크가 효과적
        chunk_overlap=160,  # 20% 오버랩으로 문맥 연결성 향상
        separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""],  # 한국어 문장 구분자 추가
        length_function=len
    )


class ElasticsearchManager:
    """Elasticsearch 검색 및 연결 관리 클래스"""
    
    @staticmethod
    def keyword_search(query: str, top_k: int = 5, index_name: str = None) -> List[Dict[str, Any]]:
        """키워드(단순 텍스트) 기반 검색"""
        # 인덱스 이름 결정 (파라미터 > 환경변수 > 기본값)
        target_index = index_name or os.getenv("INDEX_NAME") or INDEX_NAME
        
        config = ElasticsearchManager.get_connection_config()
        es = Elasticsearch(**config)
        if not es.indices.exists(index=target_index):
            return []
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "metadata.filename", "metadata.source"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        try:
            res = es.search(index=target_index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                doc = {
                    "score": hit.get("_score"),
                    "content": hit.get("_source", {}).get("text"),
                    "metadata": hit.get("_source", {}).get("metadata", {})
                }
                results.append(doc)
            return results
        except Exception as e:
            print(f"[ElasticsearchManager] 키워드 검색 오류: {e}")
            return []
    
    @staticmethod
    def get_all_indices() -> List[Dict[str, Any]]:
        """모든 Elasticsearch 인덱스 목록 조회 (상세 정보 포함)"""
        try:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            
            # 인덱스 목록과 상세 정보 조회
            indices_info = es.cat.indices(format="json", v=True)
            
            # 필요한 정보만 추출하여 정렬
            result = []
            for index_info in indices_info:
                result.append({
                    "index": index_info.get("index", ""),
                    "health": index_info.get("health", ""),
                    "status": index_info.get("status", ""),
                    "docs_count": index_info.get("docs.count", "0"),
                    "docs_deleted": index_info.get("docs.deleted", "0"),
                    "store_size": index_info.get("store.size", "0b"),
                    "pri_store_size": index_info.get("pri.store.size", "0b")
                })
            
            # 인덱스 이름으로 정렬
            result.sort(key=lambda x: x["index"])
            return result
            
        except Exception as e:
            print(f"[ElasticsearchManager] 인덱스 목록 조회 오류: {e}")
            return []
    
    @staticmethod 
    def get_index_names() -> List[str]:
        """인덱스 이름 목록만 간단하게 조회"""
        try:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            
            # 인덱스 이름만 조회
            indices = es.cat.indices(format="json", h="index")
            index_names = [idx["index"] for idx in indices]
            index_names.sort()
            return index_names
            
        except Exception as e:
            print(f"[ElasticsearchManager] 인덱스 이름 목록 조회 오류: {e}")
            return []
    
    @staticmethod
    def get_connection_config() -> Dict[str, Any]:
        """Elasticsearch 연결 설정 생성"""
        config = {
            "hosts": [ELASTICSEARCH_URL],
            "verify_certs": False,
            "ssl_show_warn": False,
            "request_timeout": 30,
            "max_retries": 3,
            "retry_on_timeout": True
        }
        
        # 인증 정보가 있으면 추가
        if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
            config["basic_auth"] = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
        
        return config
    
    @staticmethod
    def check_connection() -> Tuple[bool, str]:
        """Elasticsearch 연결 확인"""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 주 연결 설정
        main_config = ElasticsearchManager.get_connection_config()
        
        # 대체 연결 방법들
        fallback_configs = [
            {
                "hosts": ["http://localhost:9200"],
                "verify_certs": False,
                "request_timeout": 30
            },
            {
                "hosts": ["http://127.0.0.1:9200"],
                "verify_certs": False,
                "request_timeout": 30
            }
        ]
        
        # 모든 설정을 합쳐서 시도
        all_configs = [main_config] + fallback_configs
        
        for i, config in enumerate(all_configs):
            try:
                es = Elasticsearch(**config)
                if es.ping():
                    cluster_info = es.info()
                    version = cluster_info.get('version', {}).get('number', 'Unknown')
                    cluster_name = cluster_info.get('cluster_name', 'Unknown')
                    host_info = config["hosts"][0]
                    return True, f"연결 성공 ({cluster_name} v{version}) - {host_info}"
            except Exception as e:
                if i == 0:  # 주 연결 실패 시에만 에러 정보 기록
                    last_error = str(e)
                continue
        
        error_msg = f"모든 연결 방법 실패"
        if 'last_error' in locals():
            error_msg += f" (주 연결 오류: {last_error})"
        error_msg += ". Elasticsearch가 실행 중인지 확인하세요."
        
        return False, error_msg
    
    @staticmethod
    def get_safe_elasticsearch_client() -> Tuple[Elasticsearch, bool, str]:
        """안전한 Elasticsearch 클라이언트 반환"""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        configs = [
            {
                "hosts": [ELASTICSEARCH_URL],
                "verify_certs": False,
                "ssl_show_warn": False,
                "request_timeout": 30,
                "max_retries": 3,
                "retry_on_timeout": True
            },
            {
                "hosts": ["http://localhost:9200"],
                "verify_certs": False,
                "request_timeout": 30
            }
        ]
        
        for config in configs:
            try:
                es = Elasticsearch(**config)
                if es.ping():
                    return es, True, f"클라이언트 생성 성공: {config['hosts'][0]}"
            except Exception:
                continue
        
        return None, False, "Elasticsearch 클라이언트 생성 실패"
