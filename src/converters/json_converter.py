"""
DOCX를 JSON으로 변환하는 모듈
"""
import json
from typing import Tuple, Dict, Any
from datetime import datetime
from .base import BaseConverter


class JsonConverter(BaseConverter):
    """DOCX를 JSON으로 변환하는 클래스"""
    
    def convert(self) -> Tuple[bytes, str]:
        """
        DOCX를 기본 JSON으로 변환합니다.
        
        Returns:
            Tuple[bytes, str]: (JSON 파일의 바이트 데이터, MIME 타입)
        """
        # 구조화된 내용 추출
        structured_content = self.extractor.extract_structured_content()
        
        # 기본 JSON 구조 생성
        json_data = {
            "document_type": "docx_conversion",
            "conversion_timestamp": datetime.now().isoformat(),
            "content": structured_content
        }
        
        # JSON으로 직렬화
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        return json_bytes, "application/json"
    
    def convert_simple_format(self) -> Tuple[bytes, str]:
        """
        DOCX를 간단한 JSON 형식으로 변환합니다.
        
        Returns:
            Tuple[bytes, str]: (단순 JSON 파일의 바이트 데이터, MIME 타입)
        """
        # 순수 텍스트 추출
        text_content = self.extractor.extract_text_content()
        
        # 간단한 JSON 구조
        json_data = {
            "text": text_content,
            "timestamp": datetime.now().isoformat(),
            "source": "docx_conversion"
        }
        
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        return json_bytes, "application/json"
    
    def convert_detailed_format(self) -> Tuple[bytes, str]:
        """
        DOCX를 상세한 JSON 형식으로 변환합니다.
        
        Returns:
            Tuple[bytes, str]: (상세 JSON 파일의 바이트 데이터, MIME 타입)
        """
        structured_content = self.extractor.extract_structured_content()
        
        # 상세한 JSON 구조 생성
        json_data = {
            "document_info": {
                "type": "docx_document",
                "conversion_timestamp": datetime.now().isoformat(),
                "statistics": structured_content["metadata"]
            },
            "content": {
                "paragraphs": structured_content["paragraphs"],
                "tables": structured_content["tables"]
            },
            "full_text": self.extractor.extract_text_content(),
            "extraction_metadata": {
                "extractor_version": "1.0",
                "features_extracted": [
                    "paragraphs",
                    "tables", 
                    "text_content",
                    "paragraph_styles"
                ]
            }
        }
        
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        return json_bytes, "application/json"
