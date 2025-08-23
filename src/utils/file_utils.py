"""
파일 인덱싱 관련 유틸리티 함수들
"""

import os
import time
from typing import List, Tuple


def auto_index_files(file_types: List[str], index_name: str) -> None:
    """파일 자동 인덱싱 함수"""
    from core.models import ModelFactory
    from utils.elasticsearch_index import ElasticsearchIndexer
    from elasticsearch import Elasticsearch
    
    # 파일 타입 처리
    if 'all' in file_types:
        file_types = ['pdf', 'txt', 'json', 'docx']
    
    print(f"\n📁 파일 자동 인덱싱 시작 (INDEX_NAME: {index_name})")
    print(f"📋 대상 파일 타입: {', '.join(file_types)}")
    
    # data 디렉토리 확인 및 생성
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"📁 데이터 디렉토리({data_dir})가 없습니다. 생성합니다...")
        os.makedirs(data_dir, exist_ok=True)
    
    # 각 파일 타입별 디렉토리 및 파일 확인
    from utils.elasticsearch_index import ElasticsearchIndexer
    total_files = 0
    file_info = {}
    
    for file_type in file_types:
        type_dir = os.path.join(data_dir, file_type)
        if not os.path.exists(type_dir):
            print(f"📂 {file_type.upper()} 디렉토리({type_dir})가 없습니다. 생성합니다...")
            os.makedirs(type_dir, exist_ok=True)
            file_info[file_type] = []
        else:
            if file_type == 'pdf':
                files = ElasticsearchIndexer.list_pdfs(type_dir)
            elif file_type == 'txt':
                files = ElasticsearchIndexer.list_txt_files(type_dir)
            elif file_type == 'json':
                files = ElasticsearchIndexer.list_json_files(type_dir)
            elif file_type == 'docx':
                files = ElasticsearchIndexer.list_docx_files(type_dir)
            else:
                files = []
            
            file_info[file_type] = files
            total_files += len(files)
            print(f"📄 {file_type.upper()} 파일: {len(files)}개")
            for file_path in files:
                print(f"  - {os.path.basename(file_path)}")
    
    if total_files == 0:
        print("📄 인덱싱할 파일이 없습니다. 기존 문서 삭제를 진행합니다.")
    
    # 기존 인덱스 확인
    try:
        es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"), timeout=10)
        if es.indices.exists(index=index_name):
            doc_count = es.count(index=index_name).get("count", 0)
            if doc_count > 0:
                print(f"📚 기존 인덱스에 {doc_count}개 문서가 있습니다. 기존 인덱스를 삭제하고 새로 생성합니다.")
    except Exception as check_error:
        print(f"⚠️ 인덱스 확인 중 오류: {str(check_error)}, 인덱싱을 계속 진행합니다...")
    
    print(f"📄 총 {total_files}개 파일을 자동 인덱싱합니다...")
    
    try:
        class SimpleTracker:
            def track_preprocessing_stage(self, stage):
                print(f"🔄 {stage}")
            def end_preprocessing_stage(self, stage):
                print(f"✅ {stage} 완료")
        
        tracker = SimpleTracker()
        embedding_model = ModelFactory().create_embedding_model()
        indexing_start = time.time()
        
        # 통합 인덱싱 실행
        indexer = ElasticsearchIndexer()
        success, message = indexer.index_all_files(data_dir, embedding_model, tracker, file_types)
        
        indexing_time = time.time() - indexing_start
        if success:
            print(f"✅ 파일 자동 인덱싱 완료: {message} ({indexing_time:.2f}초)")
        else:
            print(f"❌ 파일 자동 인덱싱 실패: {message}")
            
    except Exception as e:
        print(f"❌ 파일 자동 인덱싱 오류: {str(e)}")


def create_safe_filename(filename: str) -> str:
    """안전한 파일명 생성"""
    import re
    # 파일명에서 한국어나 특수문자 제거하여 안전한 파일명 생성
    safe_filename = re.sub(r'[^\w\-.]', '', filename)
    
    # 빈 파일명인 경우 기본값 사용
    if not safe_filename or safe_filename.isspace():
        safe_filename = "converted_document"
    
    return safe_filename
