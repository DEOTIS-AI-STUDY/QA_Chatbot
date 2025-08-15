"""
Elasticsearch 관련 유틸리티
"""
import os
import urllib3
from typing import List, Tuple
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from core.config import ELASTICSEARCH_URL, INDEX_NAME


class ElasticsearchManager:
    """Elasticsearch 관리 클래스"""
    
    @staticmethod
    def check_connection() -> Tuple[bool, str]:
        """Elasticsearch 연결 확인 (개선된 버전)"""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 다양한 연결 방법 시도
        connection_configs = [
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
            },
            {
                "hosts": ["http://127.0.0.1:9200"],
                "verify_certs": False,
                "request_timeout": 30
            }
        ]
        
        for i, config in enumerate(connection_configs):
            try:
                es = Elasticsearch(**config)
                if es.ping():
                    cluster_info = es.info()
                    version = cluster_info.get('version', {}).get('number', 'Unknown')
                    return True, f"연결 성공 (v{version}) - 방법 {i+1}"
            except Exception:
                continue
        
        return False, "모든 연결 방법 실패. Elasticsearch가 실행 중인지 확인하세요."
    
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
    
    @staticmethod
    def list_pdfs(pdf_dir: str) -> List[str]:
        """PDF 파일 목록 반환"""
        try:
            names = os.listdir(pdf_dir)
        except FileNotFoundError:
            return []
        
        files = []
        for name in names:
            if name.lower().endswith(".pdf"):
                files.append(os.path.join(pdf_dir, name))
        return sorted(files)
    
    @staticmethod
    def index_pdfs(pdf_files: List[str], embeddings, hybrid_tracker) -> Tuple[bool, str]:
        """PDF 파일들을 Elasticsearch에 인덱싱"""
        hybrid_tracker.track_preprocessing_stage("PDF_인덱싱_시작")
        
        # 기존 인덱스 삭제
        es = Elasticsearch(ELASTICSEARCH_URL)
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
        
        all_documents = []
        
        for pdf_path in pdf_files:
            hybrid_tracker.track_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
            
            # PDF 로딩
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 텍스트 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            
            # 메타데이터 보강
            for chunk in chunks:
                chunk.metadata.update({
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "category": "PDF"
                })
            
            all_documents.extend(chunks)
            hybrid_tracker.end_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            return False, "추출된 텍스트가 없습니다."
        
        # Elasticsearch에 저장
        hybrid_tracker.track_preprocessing_stage("Elasticsearch_저장")
        try:
            # 안전한 Elasticsearch 클라이언트 확인
            es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
            if not success:
                raise Exception(f"Elasticsearch 연결 실패: {message}")
            
            # 문서 저장
            ElasticsearchStore.from_documents(
                all_documents,
                embedding=embeddings,
                es_url=ELASTICSEARCH_URL,
                index_name=INDEX_NAME
            )
            
            es_client.indices.refresh(index=INDEX_NAME)
            cnt = es_client.count(index=INDEX_NAME).get("count", 0)
            
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            
            return True, f"인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            return False, f"인덱싱 오류: {str(e)}"
