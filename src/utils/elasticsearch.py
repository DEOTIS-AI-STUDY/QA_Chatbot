"""
Elasticsearch 관련 유틸리티
"""
import os
import urllib3
import json
from typing import List, Tuple, Optional, Dict, Any
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from langchain.schema import Document
from core.config import (
    ELASTICSEARCH_URL, 
    ELASTICSEARCH_HOST, 
    ELASTICSEARCH_PORT, 
    ELASTICSEARCH_SCHEME,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    INDEX_NAME
)


class ElasticsearchManager:
    @staticmethod
    def keyword_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """키워드(단순 텍스트) 기반 검색"""
        config = ElasticsearchManager.get_connection_config()
        es = Elasticsearch(**config)
        if not es.indices.exists(index=INDEX_NAME):
            return []
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["page_content", "metadata.filename", "metadata.source"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        try:
            res = es.search(index=INDEX_NAME, body=body)
            hits = res.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                doc = {
                    "score": hit.get("_score"),
                    "content": hit.get("_source", {}).get("page_content"),
                    "metadata": hit.get("_source", {}).get("metadata", {})
                }
                results.append(doc)
            return results
        except Exception as e:
            print(f"[ElasticsearchManager] 키워드 검색 오류: {e}")
            return []
    """Elasticsearch 관리 클래스"""
    
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
        """Elasticsearch 연결 확인 (개선된 버전)"""
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
    def list_txt_files(txt_dir: str) -> List[str]:
        """TXT 파일 목록 반환"""
        try:
            names = os.listdir(txt_dir)
        except FileNotFoundError:
            return []
        
        files = []
        for name in names:
            if name.lower().endswith(".txt"):
                files.append(os.path.join(txt_dir, name))
        return sorted(files)
    
    @staticmethod
    def delete_documents_by_category(categories: List[str]) -> Tuple[bool, str]:
        """특정 카테고리의 문서들을 선택적으로 삭제"""
        try:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            
            if not es.indices.exists(index=INDEX_NAME):
                return True, "인덱스가 존재하지 않습니다."
            
            # 삭제할 카테고리들에 대한 쿼리 생성
            delete_query = {
                "query": {
                    "terms": {
                        "metadata.category.keyword": categories
                    }
                }
            }
            
            # 삭제 전 문서 수 확인
            count_before = es.count(index=INDEX_NAME).get("count", 0)
            
            # 선택적 삭제 실행
            delete_response = es.delete_by_query(
                index=INDEX_NAME,
                body=delete_query,
                refresh=True
            )
            
            deleted_count = delete_response.get("deleted", 0)
            count_after = es.count(index=INDEX_NAME).get("count", 0)
            
            return True, f"카테고리 {categories} 문서 {deleted_count}개 삭제됨 (전체: {count_before} → {count_after})"
            
        except Exception as e:
            return False, f"선택적 삭제 오류: {str(e)}"
    
    @staticmethod
    def get_existing_files_by_category(category: str) -> List[str]:
        """특정 카테고리의 기존 파일명 목록 반환"""
        try:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            
            if not es.indices.exists(index=INDEX_NAME):
                return []
            
            # 카테고리별 파일명 집계 쿼리
            agg_query = {
                "size": 0,
                "query": {
                    "term": {
                        "metadata.category.keyword": category
                    }
                },
                "aggs": {
                    "filenames": {
                        "terms": {
                            "field": "metadata.filename.keyword",
                            "size": 1000
                        }
                    }
                }
            }
            
            response = es.search(index=INDEX_NAME, body=agg_query)
            buckets = response.get("aggregations", {}).get("filenames", {}).get("buckets", [])
            
            return [bucket["key"] for bucket in buckets]
            
        except Exception as e:
            print(f"기존 파일 목록 조회 오류: {e}")
            return []
    
    @staticmethod
    def list_json_files(json_dir: str) -> List[str]:
        """JSON 파일 목록 반환"""
        try:
            names = os.listdir(json_dir)
        except FileNotFoundError:
            return []
        
        files = []
        for name in names:
            if name.lower().endswith(".json"):
                files.append(os.path.join(json_dir, name))
        return sorted(files)
    
    @staticmethod
    def index_pdfs(pdf_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """PDF 파일들을 Elasticsearch에 인덱싱 (증분 업데이트 지원)"""
        hybrid_tracker.track_preprocessing_stage("PDF_인덱싱_시작")
        
        # 증분 업데이트 모드가 아니면 기존 인덱스 완전 삭제
        if not incremental:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            if es.indices.exists(index=INDEX_NAME):
                es.indices.delete(index=INDEX_NAME)
        else:
            # 기존 PDF 카테고리 문서만 삭제
            success, message = ElasticsearchManager.delete_documents_by_category(["PDF"])
            if not success:
                hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
                return False, f"기존 PDF 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
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
            
            return True, f"PDF 인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("PDF_인덱싱_시작")
            return False, f"PDF 인덱싱 오류: {str(e)}"
    
    @staticmethod
    def index_txt_files(txt_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """TXT 파일들을 Elasticsearch에 인덱싱 (증분 업데이트 지원)"""
        hybrid_tracker.track_preprocessing_stage("TXT_인덱싱_시작")
        
        # 증분 업데이트 모드가 아니면 기존 인덱스 완전 삭제
        if not incremental:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            if es.indices.exists(index=INDEX_NAME):
                es.indices.delete(index=INDEX_NAME)
        else:
            # 기존 TXT 카테고리 문서만 삭제
            success, message = ElasticsearchManager.delete_documents_by_category(["TXT"])
            if not success:
                hybrid_tracker.end_preprocessing_stage("TXT_인덱싱_시작")
                return False, f"기존 TXT 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
        all_documents = []
        
        for txt_path in txt_files:
            hybrid_tracker.track_preprocessing_stage(f"TXT_처리_{os.path.basename(txt_path)}")
            
            try:
                # TXT 로딩
                loader = TextLoader(txt_path, encoding='utf-8')
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
                        "source": txt_path,
                        "filename": os.path.basename(txt_path),
                        "category": "TXT"
                    })
                
                all_documents.extend(chunks)
                
            except Exception as e:
                print(f"TXT 파일 처리 오류 ({txt_path}): {e}")
                continue
                
            hybrid_tracker.end_preprocessing_stage(f"TXT_처리_{os.path.basename(txt_path)}")
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage("TXT_인덱싱_시작")
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
            hybrid_tracker.end_preprocessing_stage("TXT_인덱싱_시작")
            
            return True, f"TXT 인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("TXT_인덱싱_시작")
            return False, f"TXT 인덱싱 오류: {str(e)}"
    
    @staticmethod
    def index_json_files(json_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """JSON 파일들을 Elasticsearch에 인덱싱 (증분 업데이트 지원)"""
        hybrid_tracker.track_preprocessing_stage("JSON_인덱싱_시작")
        
        # 증분 업데이트 모드가 아니면 기존 인덱스 완전 삭제
        if not incremental:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            if es.indices.exists(index=INDEX_NAME):
                es.indices.delete(index=INDEX_NAME)
        else:
            # 기존 JSON 카테고리 문서만 삭제
            success, message = ElasticsearchManager.delete_documents_by_category(["JSON"])
            if not success:
                hybrid_tracker.end_preprocessing_stage("JSON_인덱싱_시작")
                return False, f"기존 JSON 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
        all_documents = []
        
        for json_path in json_files:
            hybrid_tracker.track_preprocessing_stage(f"JSON_처리_{os.path.basename(json_path)}")
            
            try:
                # JSON 로딩
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # JSON 구조에 따라 텍스트 추출
                text_content = ElasticsearchManager._extract_text_from_json(json_data)
                
                if text_content:
                    # Document 객체 생성
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": json_path,
                            "filename": os.path.basename(json_path),
                            "category": "JSON"
                        }
                    )
                    
                    # 텍스트 분할
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents([doc])
                    
                    all_documents.extend(chunks)
                
            except Exception as e:
                print(f"JSON 파일 처리 오류 ({json_path}): {e}")
                continue
                
            hybrid_tracker.end_preprocessing_stage(f"JSON_처리_{os.path.basename(json_path)}")
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage("JSON_인덱싱_시작")
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
            hybrid_tracker.end_preprocessing_stage("JSON_인덱싱_시작")
            
            return True, f"JSON 인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("JSON_인덱싱_시작")
            return False, f"JSON 인덱싱 오류: {str(e)}"
    
    @staticmethod
    def _extract_text_from_json(json_data: Any, max_depth: int = 10) -> str:
        """JSON 데이터에서 텍스트 추출"""
        if max_depth <= 0:
            return ""
        
        text_parts = []
        
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                # 키도 텍스트로 포함
                text_parts.append(str(key))
                # 값 추출
                if isinstance(value, (dict, list)):
                    text_parts.append(ElasticsearchManager._extract_text_from_json(value, max_depth - 1))
                else:
                    text_parts.append(str(value))
        elif isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, (dict, list)):
                    text_parts.append(ElasticsearchManager._extract_text_from_json(item, max_depth - 1))
                else:
                    text_parts.append(str(item))
        else:
            text_parts.append(str(json_data))
        
        return " ".join(filter(None, text_parts))
    
    @staticmethod
    def index_all_files(data_dir: str, embeddings, hybrid_tracker, file_types: List[str] = None, incremental: bool = True) -> Tuple[bool, str]:
        """지정된 디렉토리의 모든 파일 타입을 인덱싱 (증분 업데이트 지원)"""
        if file_types is None:
            file_types = ['pdf', 'txt', 'json']
        
        hybrid_tracker.track_preprocessing_stage("전체_파일_인덱싱_시작")
        
        # 증분 업데이트 모드가 아니면 기존 인덱스 완전 삭제
        if not incremental:
            config = ElasticsearchManager.get_connection_config()
            es = Elasticsearch(**config)
            if es.indices.exists(index=INDEX_NAME):
                es.indices.delete(index=INDEX_NAME)
        else:
            # 선택된 파일 타입의 카테고리만 삭제
            categories_to_delete = [file_type.upper() for file_type in file_types]
            success, message = ElasticsearchManager.delete_documents_by_category(categories_to_delete)
            if not success:
                hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
                return False, f"기존 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
        all_documents = []
        processed_files = 0
        
        # PDF 파일 처리
        if 'pdf' in file_types:
            pdf_dir = os.path.join(data_dir, 'pdf')
            pdf_files = ElasticsearchManager.list_pdfs(pdf_dir)
            for pdf_path in pdf_files:
                hybrid_tracker.track_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)
                    
                    for chunk in chunks:
                        chunk.metadata.update({
                            "source": pdf_path,
                            "filename": os.path.basename(pdf_path),
                            "category": "PDF"
                        })
                    
                    all_documents.extend(chunks)
                    processed_files += 1
                except Exception as e:
                    print(f"PDF 파일 처리 오류 ({pdf_path}): {e}")
                hybrid_tracker.end_preprocessing_stage(f"PDF_처리_{os.path.basename(pdf_path)}")
        
        # TXT 파일 처리
        if 'txt' in file_types:
            txt_dir = os.path.join(data_dir, 'txt')
            txt_files = ElasticsearchManager.list_txt_files(txt_dir)
            for txt_path in txt_files:
                hybrid_tracker.track_preprocessing_stage(f"TXT_처리_{os.path.basename(txt_path)}")
                try:
                    loader = TextLoader(txt_path, encoding='utf-8')
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)
                    
                    for chunk in chunks:
                        chunk.metadata.update({
                            "source": txt_path,
                            "filename": os.path.basename(txt_path),
                            "category": "TXT"
                        })
                    
                    all_documents.extend(chunks)
                    processed_files += 1
                except Exception as e:
                    print(f"TXT 파일 처리 오류 ({txt_path}): {e}")
                hybrid_tracker.end_preprocessing_stage(f"TXT_처리_{os.path.basename(txt_path)}")
        
        # JSON 파일 처리
        if 'json' in file_types:
            json_dir = os.path.join(data_dir, 'json')
            json_files = ElasticsearchManager.list_json_files(json_dir)
            for json_path in json_files:
                hybrid_tracker.track_preprocessing_stage(f"JSON_처리_{os.path.basename(json_path)}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    text_content = ElasticsearchManager._extract_text_from_json(json_data)
                    
                    if text_content:
                        doc = Document(
                            page_content=text_content,
                            metadata={
                                "source": json_path,
                                "filename": os.path.basename(json_path),
                                "category": "JSON"
                            }
                        )
                        
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = splitter.split_documents([doc])
                        
                        all_documents.extend(chunks)
                        processed_files += 1
                except Exception as e:
                    print(f"JSON 파일 처리 오류 ({json_path}): {e}")
                hybrid_tracker.end_preprocessing_stage(f"JSON_처리_{os.path.basename(json_path)}")
        
        if not all_documents:
            # 파일이 없어도 카테고리 삭제는 이미 완료됨
            hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
            
            # 현재 인덱스 문서 수 확인
            try:
                es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
                if success:
                    cnt = es_client.count(index=INDEX_NAME).get("count", 0)
                    return True, f"선택된 카테고리 삭제 완료. 처리된 파일: 0개, 현재 문서 수: {cnt}"
                else:
                    return True, "선택된 카테고리 삭제 완료. 처리된 파일: 0개"
            except:
                return True, "선택된 카테고리 삭제 완료. 처리된 파일: 0개"
        
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
            hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
            
            return True, f"전체 파일 인덱싱 완료. 처리된 파일: {processed_files}개, 문서 수: {cnt}"
            
        except Exception as e:
            hybrid_tracker.end_preprocessing_stage("Elasticsearch_저장")
            hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
            return False, f"전체 파일 인덱싱 오류: {str(e)}"
