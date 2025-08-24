"""
Elasticsearch 인덱싱 전용 유틸리티
"""
import os
import json
import re
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
                        # 기본 메타데이터 설정
                        base_metadata = {
                            "source": file_path,
                            "filename": os.path.basename(file_path)
                        }
                        
                        # JSON 파일의 경우 기존 카테고리 정보 보존
                        if file_type.lower() == 'json' and 'category' in doc.metadata:
                            base_metadata["file_type"] = file_type.upper()
                            # 기존 category는 유지
                        else:
                            base_metadata["category"] = file_type.upper()
                        
                        doc.metadata.update(base_metadata)
                        
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
        """JSON 파일 처리 - 검색 최적화를 위한 세분화된 문서 분할"""
        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        documents = []
        
        # converted_index.json 형태의 구조화된 JSON인지 확인
        if isinstance(json_data, list) and len(json_data) > 0:
            for file_entry in json_data:
                if isinstance(file_entry, dict) and 'filename' in file_entry and 'data' in file_entry:
                    # 구조화된 JSON 처리
                    filename = file_entry.get('filename', 'unknown')
                    data_items = file_entry.get('data', [])
                    
                    # 검색 최적화를 위한 다층 구조 처리
                    for item in data_items:
                        if isinstance(item, dict) and item.get('content', '').strip():
                            # 각 항목을 개별 문서로 생성 (최대한 세분화)
                            title = item.get('title', '').strip()
                            heading = item.get('heading', '').strip()
                            section = item.get('section', '').strip()
                            content = item.get('content', '').strip()
                            
                            if not content:
                                continue
                            
                            # 메타데이터 구성
                            metadata = {
                                'source_filename': filename,
                                'document_type': 'structured_json',
                                'title': title if title else '기타',
                                'heading': heading if heading else '',
                                'section': section if section else '',
                                'category': ElasticsearchIndexer._extract_category(title, heading),
                                'topic': ElasticsearchIndexer._extract_topic(title, heading, section),
                                'content_type': ElasticsearchIndexer._classify_content_type(content)
                            }
                            
                            # 검색 친화적 문서 내용 구성
                            document_parts = []
                            
                            # 제목 계층 구조 추가
                            if title:
                                document_parts.append(f"대분류: {title}")
                            if heading:
                                document_parts.append(f"소분류: {heading}")
                            if section:
                                document_parts.append(f"세부항목: {section}")
                            
                            # 핵심 키워드 추출 및 추가
                            keywords = ElasticsearchIndexer._extract_keywords(content)
                            if keywords:
                                document_parts.append(f"주요 키워드: {', '.join(keywords)}")
                            
                            # 원본 내용
                            document_parts.append(f"내용: {content}")
                            
                            # 검색용 요약 생성
                            summary = ElasticsearchIndexer._generate_summary(content, title, heading, section)
                            if summary:
                                document_parts.append(f"요약: {summary}")
                            
                            document_content = '\n\n'.join(document_parts)
                            documents.append(Document(page_content=document_content, metadata=metadata))
                    
                    # 대분류별 통합 문서도 생성 (상위 레벨 검색용)
                    title_groups = {}
                    for item in data_items:
                        if isinstance(item, dict):
                            title = item.get('title', '').strip()
                            if title and item.get('content', '').strip():
                                if title not in title_groups:
                                    title_groups[title] = []
                                title_groups[title].append(item)
                    
                    for title, items in title_groups.items():
                        if len(items) > 1:  # 여러 항목이 있는 경우만 통합 문서 생성
                            content_parts = []
                            all_keywords = set()
                            
                            for item in items:
                                content = item.get('content', '').strip()
                                if content:
                                    heading = item.get('heading', '').strip()
                                    section = item.get('section', '').strip()
                                    
                                    part_header = []
                                    if heading:
                                        part_header.append(heading)
                                    if section:
                                        part_header.append(section)
                                    
                                    if part_header:
                                        content_parts.append(f"[{' - '.join(part_header)}]\n{content}")
                                    else:
                                        content_parts.append(content)
                                    
                                    # 키워드 수집
                                    keywords = ElasticsearchIndexer._extract_keywords(content)
                                    all_keywords.update(keywords)
                            
                            if content_parts:
                                consolidated_content = '\n\n'.join(content_parts)
                                metadata = {
                                    'source_filename': filename,
                                    'document_type': 'consolidated_json',
                                    'title': title,
                                    'category': ElasticsearchIndexer._extract_category(title, ''),
                                    'keywords': ', '.join(sorted(all_keywords)),
                                    'item_count': len(items)
                                }
                                
                                document_content = f"대분류: {title}\n\n주요 키워드: {', '.join(sorted(all_keywords))}\n\n통합 내용:\n{consolidated_content}"
                                documents.append(Document(page_content=document_content, metadata=metadata))
                else:
                    # 일반 JSON 처리 (기존 방식)
                    text_content = ElasticsearchIndexer._extract_text_from_json(file_entry)
                    if text_content:
                        documents.append(Document(page_content=text_content, metadata={}))
        else:
            # 일반 JSON 처리 (기존 방식)
            text_content = ElasticsearchIndexer._extract_text_from_json(json_data)
            if text_content:
                documents.append(Document(page_content=text_content, metadata={}))
        
        return documents
    
    @staticmethod
    def _extract_category(title: str, heading: str) -> str:
        """대분류 카테고리 추출"""
        text = f"{title} {heading}".lower()
        
        if '기본업무' in text or '절차' in text:
            return '업무절차'
        elif '소비자' in text or '가이드' in text:
            return '이용가이드'
        elif '발급' in text:
            return '카드발급'
        elif '이용한도' in text:
            return '이용한도'
        elif '연체' in text or '추심' in text:
            return '연체관리'
        elif '부가서비스' in text:
            return '부가서비스'
        else:
            return '기타'
    
    @staticmethod
    def _extract_topic(title: str, heading: str, section: str) -> str:
        """세부 주제 추출"""
        parts = [p for p in [title, heading, section] if p.strip()]
        return ' - '.join(parts) if parts else '기타'
    
    @staticmethod
    def _classify_content_type(content: str) -> str:
        """내용 유형 분류"""
        content_lower = content.lower()
        
        if '절차' in content_lower or '단계' in content_lower:
            return '절차안내'
        elif '기준' in content_lower or '조건' in content_lower:
            return '기준정보'
        elif '방법' in content_lower or '활용' in content_lower:
            return '이용방법'
        elif '구분' in content_lower or '종류' in content_lower:
            return '분류정보'
        elif '주의' in content_lower or '금지' in content_lower:
            return '주의사항'
        else:
            return '일반정보'
    
    @staticmethod
    def _extract_keywords(content: str) -> List[str]:
        """내용에서 주요 키워드 추출"""
        import re
        
        # 금융 관련 주요 키워드 패턴
        keyword_patterns = [
            r'신용카드|체크카드|직불카드|선불카드',
            r'카드발급|발급기준|발급절차',
            r'이용한도|결제능력|신용등급',
            r'연체|추심|법적조치',
            r'부가서비스|포인트|할부',
            r'연회비|수수료|이자',
            r'가족카드|법인카드|복지카드',
            r'VISA|Master|AMEX|JCB',
            r'가처분소득|신용평점|연체정보'
        ]
        
        keywords = []
        for pattern in keyword_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.extend(matches)
        
        # 중요한 숫자나 비율 정보
        number_patterns = re.findall(r'\d+(?:\.\d+)?%|\d+만원|\d+개월|\d+일', content)
        keywords.extend(number_patterns)
        
        # 중복 제거 및 정렬
        return sorted(list(set(keywords)))
    
    @staticmethod
    def _generate_summary(content: str, title: str, heading: str, section: str) -> str:
        """내용 요약 생성"""
        # 첫 문장이나 핵심 문장 추출
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10 and len(first_sentence) < 200:
                return first_sentence
        
        # 제목 정보 기반 요약
        context_parts = [p for p in [title, heading, section] if p.strip()]
        if context_parts:
            return f"{' - '.join(context_parts)}에 대한 정보"
        
        return ""
    
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
