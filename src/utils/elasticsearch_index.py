"""
Elasticsearch 인덱싱 전용 유틸리티
"""
import os
import json
from typing import List, Tuple, Any
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import ElasticsearchStore
from langchain.schema import Document
from docx import Document as DocxDocument
from core.config import ELASTICSEARCH_URL, INDEX_NAME
from .elasticsearch import ElasticsearchManager, get_optimized_text_splitter


class ElasticsearchIndexer:
    """Elasticsearch 인덱싱 전용 클래스"""
    
    @staticmethod
    def delete_documents_by_category(categories: List[str]) -> Tuple[bool, str]:
        """특정 카테고리의 문서들을 선택적으로 삭제"""
        try:
            from utils.elasticsearch import ElasticsearchManager
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
            from utils.elasticsearch import ElasticsearchManager
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
    
    # 파일 목록 조회 함수들 (인덱싱에서 사용)
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
    def list_docx_files(docx_dir: str) -> List[str]:
        """DOCX 파일 목록 반환"""
        try:
            names = os.listdir(docx_dir)
        except FileNotFoundError:
            return []
        
        files = []
        for name in names:
            if name.lower().endswith((".docx", ".doc")):
                files.append(os.path.join(docx_dir, name))
        return sorted(files)
    
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
        """PDF 파일들을 Elasticsearch에 인덱싱"""
        return ElasticsearchIndexer._index_files_by_type(
            files=pdf_files,
            file_type="PDF",
            embeddings=embeddings,
            hybrid_tracker=hybrid_tracker,
            incremental=incremental,
            loader_func=lambda path: PyPDFLoader(path).load()
        )
    
    @staticmethod
    def index_txt_files(txt_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """TXT 파일들을 Elasticsearch에 인덱싱"""
        return ElasticsearchIndexer._index_files_by_type(
            files=txt_files,
            file_type="TXT",
            embeddings=embeddings,
            hybrid_tracker=hybrid_tracker,
            incremental=incremental,
            loader_func=lambda path: TextLoader(path, encoding='utf-8').load()
        )
    
    @staticmethod
    def index_json_files(json_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """JSON 파일들을 Elasticsearch에 인덱싱"""
        def json_loader(path):
            with open(path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            text_content = ElasticsearchIndexer._extract_text_from_json(json_data)
            if text_content:
                return [Document(
                    page_content=text_content,
                    metadata={"source": path, "filename": os.path.basename(path), "category": "JSON"}
                )]
            return []
        
        return ElasticsearchIndexer._index_files_by_type(
            files=json_files,
            file_type="JSON",
            embeddings=embeddings,
            hybrid_tracker=hybrid_tracker,
            incremental=incremental,
            loader_func=json_loader
        )
    
    @staticmethod
    def index_docx_files(docx_files: List[str], embeddings, hybrid_tracker, incremental: bool = True) -> Tuple[bool, str]:
        """DOCX 파일들을 Elasticsearch에 인덱싱"""
        def docx_loader(path):
            try:
                # DocParser 사용 (있는 경우)
                from utils.docparser import DocParser
                parser = DocParser(path)
                text_content = parser.get_text()
                
                docs = [Document(
                    page_content=text_content,
                    metadata={"source": path, "filename": os.path.basename(path), "category": "DOCX"}
                )]
                
                # 표 내용 추가
                table_blocks = parser.get_tables()
                for table_content in table_blocks:
                    docs.append(Document(
                        page_content=table_content,
                        metadata={"source": path, "filename": os.path.basename(path), "category": "DOCX"}
                    ))
                
                return docs
            except ImportError:
                # DocParser가 없으면 기본 python-docx 사용
                return ElasticsearchIndexer._docx_basic_loader(path)
        
        return ElasticsearchIndexer._index_files_by_type(
            files=docx_files,
            file_type="DOCX",
            embeddings=embeddings,
            hybrid_tracker=hybrid_tracker,
            incremental=incremental,
            loader_func=docx_loader
        )
    
    @staticmethod
    def _docx_basic_loader(path: str) -> List[Document]:
        """기본 DOCX 로더 (python-docx 사용)"""
        doc = DocxDocument(path)
        text_content = []
        
        # 단락 텍스트 추출
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # 표 텍스트 추출
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_content.append(cell.text)
        
        if text_content:
            full_text = "\n".join(text_content)
            return [Document(
                page_content=full_text,
                metadata={"source": path, "filename": os.path.basename(path), "category": "DOCX"}
            )]
        return []
    
    @staticmethod
    def _index_files_by_type(files: List[str], file_type: str, embeddings, hybrid_tracker, 
                           incremental: bool, loader_func) -> Tuple[bool, str]:
        """파일 타입별 통합 인덱싱 로직"""
        stage_name = f"{file_type}_인덱싱_시작"
        hybrid_tracker.track_preprocessing_stage(stage_name)
        
        # 증분 업데이트 처리
        if not incremental:
            ElasticsearchIndexer._delete_entire_index()
        else:
            success, message = ElasticsearchIndexer.delete_documents_by_category([file_type])
            if not success:
                hybrid_tracker.end_preprocessing_stage(stage_name)
                return False, f"기존 {file_type} 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
        all_documents = []
        
        # 파일별 처리
        for file_path in files:
            file_stage = f"{file_type}_처리_{os.path.basename(file_path)}"
            hybrid_tracker.track_preprocessing_stage(file_stage)
            
            try:
                docs = loader_func(file_path)
                
                # 텍스트 분할 (표는 분할하지 않음)
                splitter = get_optimized_text_splitter()
                for doc in docs:
                    if "표" in doc.page_content or len(doc.page_content.split('\n')) > 5:
                        # 표이거나 여러 줄인 경우 분할하지 않음
                        all_documents.append(doc)
                    else:
                        # 일반 텍스트는 분할
                        chunks = splitter.split_documents([doc])
                        all_documents.extend(chunks)
                
                # 메타데이터 보강
                for doc in all_documents[-len(docs):]:  # 방금 추가된 문서들만
                    doc.metadata.update({
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                        "category": file_type
                    })
                
            except Exception as e:
                print(f"{file_type} 파일 처리 오류 ({file_path}): {e}")
                continue
            finally:
                hybrid_tracker.end_preprocessing_stage(file_stage)
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage(stage_name)
            return False, "추출된 텍스트가 없습니다."
        
        # Elasticsearch에 저장
        return ElasticsearchIndexer._save_to_elasticsearch(all_documents, embeddings, hybrid_tracker, stage_name, file_type)
    
    @staticmethod
    def _delete_entire_index():
        """전체 인덱스 삭제"""
        config = ElasticsearchManager.get_connection_config()
        es = Elasticsearch(**config)
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
    
    @staticmethod
    def _save_to_elasticsearch(documents: List[Document], embeddings, hybrid_tracker, stage_name: str, file_type: str) -> Tuple[bool, str]:
        """Elasticsearch에 문서 저장"""
        save_stage = "Elasticsearch_저장"
        hybrid_tracker.track_preprocessing_stage(save_stage)
        
        try:
            es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
            if not success:
                raise Exception(f"Elasticsearch 연결 실패: {message}")
            
            ElasticsearchStore.from_documents(
                documents,
                embedding=embeddings,
                es_url=ELASTICSEARCH_URL,
                index_name=INDEX_NAME
            )
            
            es_client.indices.refresh(index=INDEX_NAME)
            cnt = es_client.count(index=INDEX_NAME).get("count", 0)
            
            return True, f"{file_type} 인덱싱 완료. 문서 수: {cnt}"
            
        except Exception as e:
            return False, f"{file_type} 인덱싱 오류: {str(e)}"
        finally:
            hybrid_tracker.end_preprocessing_stage(save_stage)
            hybrid_tracker.end_preprocessing_stage(stage_name)
    
    @staticmethod
    def _extract_text_from_json(json_data: Any, max_depth: int = 10) -> str:
        """JSON 데이터에서 텍스트 추출"""
        if max_depth <= 0:
            return ""
        
        text_parts = []
        
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                text_parts.append(str(key))
                if isinstance(value, (dict, list)):
                    text_parts.append(ElasticsearchIndexer._extract_text_from_json(value, max_depth - 1))
                else:
                    text_parts.append(str(value))
        elif isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, (dict, list)):
                    text_parts.append(ElasticsearchIndexer._extract_text_from_json(item, max_depth - 1))
                else:
                    text_parts.append(str(item))
        else:
            text_parts.append(str(json_data))
        
        return " ".join(filter(None, text_parts))
    
    @staticmethod
    def index_all_files(data_dir: str, embeddings, hybrid_tracker, file_types: List[str] = None, incremental: bool = True) -> Tuple[bool, str]:
        """모든 파일 타입을 한 번에 인덱싱"""
        if file_types is None:
            file_types = ['pdf', 'txt', 'json', 'docx']
        
        hybrid_tracker.track_preprocessing_stage("전체_파일_인덱싱_시작")
        
        # 증분 업데이트 처리
        if not incremental:
            ElasticsearchIndexer._delete_entire_index()
        else:
            categories_to_delete = [ft.upper() for ft in file_types]
            success, message = ElasticsearchIndexer.delete_documents_by_category(categories_to_delete)
            if not success:
                hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
                return False, f"기존 문서 삭제 실패: {message}"
            print(f"📚 {message}")
        
        all_documents = []
        processed_files = 0
        
        # 파일 타입별 처리 매핑
        file_processors = {
            'pdf': (ElasticsearchIndexer.list_pdfs, lambda path: PyPDFLoader(path).load()),
            'txt': (ElasticsearchIndexer.list_txt_files, lambda path: TextLoader(path, encoding='utf-8').load()),
            'json': (ElasticsearchIndexer.list_json_files, ElasticsearchIndexer._process_json_file),
            'docx': (ElasticsearchIndexer.list_docx_files, ElasticsearchIndexer._process_docx_file)
        }
        
        for file_type in file_types:
            if file_type not in file_processors:
                continue
                
            list_func, loader_func = file_processors[file_type]
            type_dir = os.path.join(data_dir, file_type)
            files = list_func(type_dir)
            
            for file_path in files:
                stage_name = f"{file_type.upper()}_처리_{os.path.basename(file_path)}"
                hybrid_tracker.track_preprocessing_stage(stage_name)
                
                try:
                    docs = loader_func(file_path)
                    splitter = get_optimized_text_splitter()
                    
                    for doc in docs:
                        doc.metadata.update({
                            "source": file_path,
                            "filename": os.path.basename(file_path),
                            "category": file_type.upper()
                        })
                        
                        # 표가 아닌 경우만 분할
                        if "표" not in doc.page_content:
                            chunks = splitter.split_documents([doc])
                            all_documents.extend(chunks)
                        else:
                            all_documents.append(doc)
                    
                    processed_files += 1
                    
                except Exception as e:
                    print(f"{file_type.upper()} 파일 처리 오류 ({file_path}): {e}")
                finally:
                    hybrid_tracker.end_preprocessing_stage(stage_name)
        
        if not all_documents:
            hybrid_tracker.end_preprocessing_stage("전체_파일_인덱싱_시작")
            try:
                es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
                cnt = es_client.count(index=INDEX_NAME).get("count", 0) if success else 0
                return True, f"선택된 카테고리 삭제 완료. 처리된 파일: 0개, 현재 문서 수: {cnt}"
            except:
                return True, "선택된 카테고리 삭제 완료. 처리된 파일: 0개"
        
        # Elasticsearch에 저장
        result, message = ElasticsearchIndexer._save_to_elasticsearch(
            all_documents, embeddings, hybrid_tracker, "전체_파일_인덱싱_시작", "전체"
        )
        
        if result:
            return True, f"전체 파일 인덱싱 완료. 처리된 파일: {processed_files}개, 총 문서 : {message.split('문서 수: ')[1] if '문서 수: ' in message else ''}"
        else:
            return False, message
    
    @staticmethod
    def _process_json_file(path: str) -> List[Document]:
        """JSON 파일 처리"""
        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        text_content = ElasticsearchIndexer._extract_text_from_json(json_data)
        if text_content:
            return [Document(page_content=text_content, metadata={})]
        return []
    
    @staticmethod
    def _process_docx_file(path: str) -> List[Document]:
        """DOCX 파일 처리"""
        try:
            from utils.docparser import DocParser
            parser = DocParser(path)
            text_content = parser.get_text()
            
            docs = [Document(page_content=text_content, metadata={})]
            
            # 표 내용 추가
            table_blocks = parser.get_tables()
            for table_content in table_blocks:
                docs.append(Document(page_content=table_content, metadata={}))
            
            return docs
        except ImportError:
            return ElasticsearchIndexer._docx_basic_loader(path)
