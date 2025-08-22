"""
DOCX를 TXT로 변환하는 모듈
"""
from typing import Tuple
from .base import BaseConverter


class TxtConverter(BaseConverter):
    """DOCX를 TXT로 변환하는 클래스"""
    
    def convert(self) -> Tuple[bytes, str]:
        """
        DOCX를 TXT로 변환합니다.
        
        Returns:
            Tuple[bytes, str]: (TXT 파일의 바이트 데이터, MIME 타입)
        """
        try:
            # 순수 텍스트 추출
            text_content = self.extractor.extract_text_content()
            
            # 텍스트가 비어있는 경우 처리
            if not text_content.strip():
                text_content = "문서에서 텍스트를 찾을 수 없습니다."
            
            # UTF-8로 안전하게 인코딩
            txt_bytes = text_content.encode('utf-8', errors='replace')
            
            return txt_bytes, "text/plain; charset=utf-8"
            
        except Exception as e:
            print(f"TXT 변환 오류: {e}")
            # 오류 발생 시 안전한 텍스트 반환
            error_text = f"텍스트 변환 중 오류가 발생했습니다: {str(e)}"
            txt_bytes = error_text.encode('utf-8', errors='replace')
            return txt_bytes, "text/plain; charset=utf-8"
    
    def convert_with_formatting(self) -> Tuple[bytes, str]:
        """
        DOCX를 서식이 포함된 TXT로 변환합니다.
        
        Returns:
            Tuple[bytes, str]: (포맷된 TXT 파일의 바이트 데이터, MIME 타입)
        """
        try:
            structured_content = self.extractor.extract_structured_content()
            
            formatted_text = []
            formatted_text.append("=" * 50)
            formatted_text.append("DOCX 문서 변환 결과")
            formatted_text.append("=" * 50)
            formatted_text.append("")
            
            # 단락 처리
            if structured_content["paragraphs"]:
                formatted_text.append("📄 문서 내용:")
                formatted_text.append("-" * 30)
                
                for para in structured_content["paragraphs"]:
                    if para["style"] != "Normal":
                        formatted_text.append(f"[{para['style']}] {para['text']}")
                    else:
                        formatted_text.append(para["text"])
                    formatted_text.append("")
            
            # 표 처리
            if structured_content["tables"]:
                formatted_text.append("📊 표 데이터:")
                formatted_text.append("-" * 30)
                
                for i, table in enumerate(structured_content["tables"]):
                    formatted_text.append(f"표 {i + 1}:")
                    for row in table["rows"]:
                        formatted_text.append(" | ".join(row))
                    formatted_text.append("")
            
            # 메타데이터
            metadata = structured_content["metadata"]
            formatted_text.append("📈 문서 정보:")
            formatted_text.append("-" * 30)
            formatted_text.append(f"총 단락 수: {metadata['total_paragraphs']}")
            formatted_text.append(f"총 표 수: {metadata['total_tables']}")
            formatted_text.append(f"총 문자 수: {metadata['total_characters']}")
            
            full_text = "\n".join(formatted_text)
            
            # 안전한 UTF-8 인코딩
            txt_bytes = full_text.encode('utf-8', errors='replace')
            
            return txt_bytes, "text/plain; charset=utf-8"
            
        except Exception as e:
            print(f"포맷된 TXT 변환 오류: {e}")
            # 오류 발생 시 기본 변환으로 폴백
            return self.convert()
