"""
개선된 DOCX → index.json 변환기
원본 index.json 구조를 더 정확하게 재현
"""
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from docx import Document

# 로컬 import 처리
try:
    from .docx_utils import ContentExtractor
except ImportError:
    try:
        from docx_utils import ContentExtractor
    except ImportError:
        # 직접 구현
        class ContentExtractor:
            def extract_all_content(self, doc_path: str) -> List[Dict[str, Any]]:
                doc = Document(doc_path)
                content_items = []
                
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        content_items.append({
                            'type': 'paragraph',
                            'content': text
                        })
                
                return content_items


class ImprovedPatternMatcher:
    """개선된 패턴 매칭 클래스"""
    
    def __init__(self):
        # 우선순위 순서대로 패턴 정의
        self.title_patterns = [
            r'^\[([^\]]+)\]$',           # [제목]
            r'^【([^】]+)】$',             # 【제목】
            r'^#+\s*(.+)$',             # # 제목
        ]
        
        # heading 패턴 - 숫자. 형태만 (더 엄격하게)
        self.heading_patterns = [
            r'^(\d+\.\s+[^)]+)$',        # 1. 제목 (괄호 없이)
            r'^(제\d+[조항절]\s*.+)$',    # 제1조 형태
        ]
        
        # section 패턴 - 더 구체적으로
        self.section_patterns = [
            r'^(\d+-\d+\)\s*.+)$',       # 1-1) 형태
            r'^(\d+\)\s*.+)$',           # 1) 형태  
            r'^([①-⑳]\s*.+)$',          # ① 형태
            r'^([가-힣]\)\s*.+)$',        # 가) 형태
        ]
    
    def match_pattern(self, text: str) -> Tuple[str, Optional[str]]:
        """패턴 매칭 with 우선순위"""
        text = text.strip()
        
        # 1. Title 패턴 확인 (가장 높은 우선순위)
        for pattern in self.title_patterns:
            match = re.match(pattern, text)
            if match:
                return "title", text
        
        # 2. Section 패턴 확인 (Heading보다 먼저)
        for pattern in self.section_patterns:
            match = re.match(pattern, text)
            if match:
                return "section", text
        
        # 3. Heading 패턴 확인
        for pattern in self.heading_patterns:
            match = re.match(pattern, text)
            if match:
                return "heading", text
        
        # 4. 기본값
        return "text", None


class ImprovedIndexJSONGenerator:
    """개선된 index.json 생성기"""
    
    def __init__(self):
        self.pattern_matcher = ImprovedPatternMatcher()
        self.content_extractor = ContentExtractor()
    
    def generate_index_json(self, docx_files: List[str], output_path: str = None) -> List[Dict[str, Any]]:
        """DOCX 파일들을 index.json 형태로 변환"""
        result = []
        
        for docx_file in docx_files:
            if not os.path.exists(docx_file):
                print(f"파일을 찾을 수 없습니다: {docx_file}")
                continue
            
            filename = os.path.basename(docx_file)
            print(f"처리 중: {filename}")
            
            # DOCX 파일 파싱
            file_data = self._parse_docx_file(docx_file)
            
            result.append({
                "filename": filename,
                "data": file_data
            })
        
        # 파일로 저장
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"결과 저장됨: {output_path}")
        
        return result
    
    def _parse_docx_file(self, docx_path: str) -> List[Dict[str, Any]]:
        """개별 DOCX 파일을 파싱하여 데이터 추출"""
        # 컨텐츠 추출
        content_items = self.content_extractor.extract_all_content(docx_path)
        
        parsed_data = []
        current_title = ""
        current_heading = ""
        current_section = ""
        content_buffer = []
        
        for item in content_items:
            item_type = item.get('type')
            
            # 에러 타입 처리
            if item_type == 'error':
                print(f"오류 발생: {item.get('content', '')}")
                continue
            
            # 표 타입 처리 - 마크다운 변환된 내용을 content_buffer에 추가
            if item_type == 'table':
                table_content = item.get('content', '')
                if table_content:
                    # 표 데이터를 텍스트 흐름에 포함 (별도 객체 생성하지 않음)
                    content_buffer.append(f"\n\n**[표]**\n\n{table_content}\n")
                continue
            
            # 단락 타입만 처리
            if item_type != 'paragraph':
                continue
            
            text = item['content'].strip()
            if not text:
                continue
            
            # 패턴 분석
            pattern_type, matched = self.pattern_matcher.match_pattern(text)
            
            if pattern_type == 'title':
                # 이전 컨텐츠 저장
                if content_buffer or current_title or current_heading or current_section:
                    self._save_content_block(parsed_data, current_title, current_heading, current_section, content_buffer)
                
                # 새로운 제목으로 설정
                current_title = text
                current_heading = ""
                current_section = ""
                content_buffer = []
                
            elif pattern_type == 'heading':
                # 이전 컨텐츠 저장
                if content_buffer or current_heading or current_section:
                    self._save_content_block(parsed_data, current_title, current_heading, current_section, content_buffer)
                
                # 새로운 헤딩 설정
                current_heading = text
                current_section = ""
                content_buffer = []
                
            elif pattern_type == 'section':
                # 이전 컨텐츠 저장
                if content_buffer or current_section:
                    self._save_content_block(parsed_data, current_title, current_heading, current_section, content_buffer)
                
                # 새로운 섹션 설정
                current_section = text
                content_buffer = []
                
            else:
                # 일반 텍스트는 버퍼에 추가
                content_buffer.append(text)
        
        # 마지막 남은 컨텐츠 저장
        if content_buffer or current_title or current_heading or current_section:
            self._save_content_block(parsed_data, current_title, current_heading, current_section, content_buffer)
        
        return parsed_data
    
    def _save_content_block(self, parsed_data: List[Dict[str, Any]], title: str, heading: str, section: str, content_lines: List[str]):
        """컨텐츠 블록을 저장"""
        # 계층 구조 생성
        hierarchy_parts = []
        
        if title:
            hierarchy_parts.append(title)
        if heading:
            hierarchy_parts.append(heading)
        if section:
            hierarchy_parts.append(section)
        
        # 실제 컨텐츠 추가
        if content_lines:
            content_text = '\n'.join(content_lines).strip()
            if content_text:
                hierarchy_parts.append(content_text)
        
        # 최종 컨텐츠 생성
        final_content = '\n'.join(hierarchy_parts) if hierarchy_parts else ""
        
        # 표 데이터 포함 여부 확인
        has_table = bool(final_content and '**[표]**' in final_content)
        
        # 데이터 추가 (빈 내용이라도 구조가 있으면 추가)
        if final_content or title or heading or section:
            data_block = {
                "title": title,
                "heading": heading, 
                "section": section,
                "content": final_content
            }
            
            # 표가 있는 경우 hasTable 속성 추가
            if has_table:
                data_block["hasTable"] = True
                
            parsed_data.append(data_block)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def generate_improved_index(directory_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    개선된 방식으로 index.json 생성
    """
    # DOCX 파일 찾기
    docx_files = []
    if os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.docx') and not filename.startswith('~'):
                docx_files.append(os.path.join(directory_path, filename))
    else:
        print(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        return []
    
    if not docx_files:
        print(f"DOCX 파일을 찾을 수 없습니다: {directory_path}")
        return []
    
    print(f"발견된 DOCX 파일: {len(docx_files)}개")
    for file in docx_files:
        print(f"  - {os.path.basename(file)}")
    
    # 개선된 생성기 실행
    generator = ImprovedIndexJSONGenerator()
    result = generator.generate_index_json(docx_files, output_file)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python improved_index_generator.py <docx_directory> [output_file]")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not output_file:
        output_file = os.path.join(directory, "improved_index.json")
    
    result = generate_improved_index(directory, output_file)
    print(f"\n완료! 총 {sum(len(item['data']) for item in result)}개의 데이터 항목이 생성되었습니다.")
