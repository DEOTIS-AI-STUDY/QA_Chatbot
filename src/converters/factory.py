"""
파일 변환 서비스 팩토리
"""
from typing import Tuple, Optional
from .base import DocxExtractor
from .txt_converter import TxtConverter
from .json_converter import JsonConverter
from .pdf_converter import PdfConverter


class FileConverterFactory:
    """파일 변환 팩토리 클래스"""
    
    @staticmethod
    def convert_docx(
        docx_content: bytes,
        output_format: str,
        conversion_type: str = "default"
    ) -> Tuple[bytes, str, str]:
        """
        DOCX 파일을 지정된 형식으로 변환합니다.
        
        Args:
            docx_content: DOCX 파일의 바이트 내용
            output_format: 출력 형식 ('txt', 'json', 'pdf')
            conversion_type: 변환 타입 ('default', 'simple', 'detailed', 'formatted')
        
        Returns:
            Tuple[bytes, str, str]: (변환된 파일 바이트, MIME 타입, 파일 확장자)
        """
        # DOCX 추출기 생성
        extractor = DocxExtractor(docx_content=docx_content)
        
        if output_format.lower() == 'txt':
            converter = TxtConverter(extractor)
            if conversion_type == "formatted":
                content_bytes, mime_type = converter.convert_with_formatting()
            else:
                content_bytes, mime_type = converter.convert()
            return content_bytes, mime_type, "txt"
        
        elif output_format.lower() == 'json':
            converter = JsonConverter(extractor)
            if conversion_type == "simple":
                content_bytes, mime_type = converter.convert_simple_format()
            elif conversion_type == "detailed":
                content_bytes, mime_type = converter.convert_detailed_format()
            else:
                content_bytes, mime_type = converter.convert()
            return content_bytes, mime_type, "json"
        
        elif output_format.lower() == 'pdf':
            converter = PdfConverter(extractor)
            # FPDF2 기반 변환기는 항상 동일한 convert() 메서드 사용
            content_bytes, mime_type = converter.convert()
            return content_bytes, mime_type, "pdf"
        
        else:
            raise ValueError(f"지원하지 않는 출력 형식입니다: {output_format}")
    
    @staticmethod
    def get_supported_formats() -> list:
        """지원하는 출력 형식 목록을 반환합니다."""
        return ["txt", "json", "pdf"]
    
    @staticmethod
    def get_conversion_types(output_format: str) -> list:
        """특정 출력 형식에서 지원하는 변환 타입 목록을 반환합니다."""
        conversion_types = {
            "txt": ["default", "formatted"],
            "json": ["default", "simple", "detailed"],
            "pdf": ["default"]
        }
        return conversion_types.get(output_format.lower(), ["default"])
