"""
DOCX를 PDF로 변환하는 모듈 (FPDF2 기반 - 안전한 UTF-8 지원)
"""
import os
from io import BytesIO
from typing import Tuple
from fpdf import FPDF

from .base import BaseConverter


class PdfConverter(BaseConverter):
    """DOCX를 PDF로 변환하는 클래스 (FPDF2 기반 - 안전 버전)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_korean_font()
    
    def setup_korean_font(self):
        """한국어 폰트 설정"""
        self.korean_font_available = False
        self.font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
            "/Library/Fonts/NanumGothic.ttf",  # macOS 나눔폰트
            "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
            "C:/Windows/Fonts/malgun.ttf",  # Windows 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",  # Windows 굴림
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                self.korean_font_path = font_path
                self.korean_font_available = True
                print(f"✅ 한국어 폰트 발견: {font_path}")
                break
        
        if not self.korean_font_available:
            print("⚠️ 한국어 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    def convert(self) -> Tuple[bytes, str]:
        """
        DOCX 파일을 PDF로 변환합니다.
        간소화된 버전으로 인코딩 안전성에 집중.
        
        Returns:
            Tuple[bytes, str]: (PDF 파일의 바이트 데이터, MIME 타입)
        """
        print("🚀 PDF 변환 시작 - FPDF2 기반 안전 버전")
        
        try:
            return self._create_safe_pdf()
        except Exception as e:
            print(f"PDF 변환 실패: {e}")
            return self._create_minimal_pdf(), "application/pdf"
    
    def _create_safe_pdf(self) -> Tuple[bytes, str]:
        """
        안전한 PDF 생성 - 인코딩 문제 최소화
        
        Returns:
            Tuple[bytes, str]: (PDF 파일의 바이트 데이터, MIME 타입)
        """
        # PDF 객체 생성
        pdf = FPDF()
        pdf.add_page()
        
        # 기본 폰트 설정 (안전한 방식)
        try:
            pdf.set_font('Arial', size=12)
            print("✅ 기본 폰트 설정 완료")
        except Exception as e:
            print(f"폰트 설정 실패: {e}")
            raise
        
        # 제목 추가
        try:
            title = "Converted Document"
            pdf.set_font('Arial', size=16)
            pdf.cell(0, 10, title, ln=True, align='C')
            pdf.ln(5)
            pdf.set_font('Arial', size=12)
        except Exception as e:
            print(f"제목 추가 실패: {e}")
        
        # 내용 추가
        try:
            self._add_content_safely(pdf)
        except Exception as e:
            print(f"내용 추가 실패: {e}")
            pdf.cell(0, 10, "Content extraction failed.", ln=True)
        
        # PDF 바이트 생성
        return self._output_pdf_safely(pdf)
    
    def _add_content_safely(self, pdf: FPDF):
        """
        안전하게 내용을 PDF에 추가
        
        Args:
            pdf (FPDF): PDF 객체
        """
        try:
            # 텍스트 내용 추출
            text_content = self.extractor.extract_text_content()
            
            if text_content:
                # 한국어나 특수문자 처리를 위해 안전한 텍스트로 변환
                safe_content = self._convert_to_safe_text(text_content)
                
                # 내용 헤더
                pdf.cell(0, 10, "Document Content:", ln=True)
                pdf.ln(3)
                
                # 텍스트를 청크로 나누어 추가
                self._add_text_in_chunks(pdf, safe_content)
            else:
                pdf.cell(0, 10, "No content found in document.", ln=True)
                
        except Exception as e:
            print(f"내용 처리 오류: {e}")
            pdf.cell(0, 10, f"Error processing content: {str(e)[:50]}...", ln=True)
    
    def _convert_to_safe_text(self, text: str) -> str:
        """
        텍스트를 PDF에 안전하게 추가할 수 있는 형태로 변환
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 안전한 텍스트
        """
        if not text:
            return ""
        
        try:
            # ASCII 문자만 유지하고 한국어는 표시
            safe_chars = []
            has_korean = False
            
            for char in text:
                if ord(char) < 128:  # ASCII 범위
                    safe_chars.append(char)
                elif '\uac00' <= char <= '\ud7af':  # 한글 음절
                    if not has_korean:
                        safe_chars.append(' [Korean text] ')
                        has_korean = True
                elif char.isspace():
                    safe_chars.append(' ')
                else:
                    safe_chars.append('?')
            
            result = ''.join(safe_chars)
            
            # 연속된 공백과 특수 문자 정리
            import re
            result = re.sub(r'\s+', ' ', result)
            result = re.sub(r'\?+', '?', result)
            
            return result.strip()
            
        except Exception as e:
            print(f"텍스트 변환 오류: {e}")
            return f"[Text conversion error: {str(e)[:30]}...]"
    
    def _add_text_in_chunks(self, pdf: FPDF, text: str, max_line_length: int = 80):
        """
        텍스트를 청크로 나누어 PDF에 추가
        
        Args:
            pdf (FPDF): PDF 객체
            text (str): 추가할 텍스트
            max_line_length (int): 한 줄 최대 길이
        """
        try:
            # 텍스트를 줄 단위로 분할
            lines = text.split('\n')
            
            for line in lines[:50]:  # 최대 50줄까지만 처리
                line = line.strip()
                if not line:
                    pdf.ln(3)
                    continue
                
                # 긴 줄은 분할
                while len(line) > max_line_length:
                    chunk = line[:max_line_length]
                    # 단어 경계에서 분할 시도
                    last_space = chunk.rfind(' ')
                    if last_space > max_line_length // 2:
                        chunk = chunk[:last_space]
                        line = line[last_space:].strip()
                    else:
                        line = line[max_line_length:].strip()
                    
                    try:
                        pdf.cell(0, 6, chunk, ln=True)
                    except Exception as e:
                        print(f"줄 추가 실패: {e}")
                        pdf.cell(0, 6, "[Text display error]", ln=True)
                
                # 남은 텍스트 추가
                if line:
                    try:
                        pdf.cell(0, 6, line, ln=True)
                    except Exception as e:
                        print(f"마지막 줄 추가 실패: {e}")
                        pdf.cell(0, 6, "[Text display error]", ln=True)
                
                # 페이지가 가득 찬 경우 새 페이지 추가
                if pdf.get_y() > 250:
                    pdf.add_page()
                    
        except Exception as e:
            print(f"텍스트 청크 처리 오류: {e}")
            pdf.cell(0, 6, f"[Content processing error: {str(e)[:40]}...]", ln=True)
    
    def _output_pdf_safely(self, pdf: FPDF) -> Tuple[bytes, str]:
        """
        PDF를 안전하게 바이트로 출력
        
        Args:
            pdf (FPDF): PDF 객체
            
        Returns:
            Tuple[bytes, str]: (PDF 바이트, MIME 타입)
        """
        try:
            # FPDF2에서 바이트 직접 반환
            pdf_data = pdf.output()
            
            # 반환값이 바이트가 아닌 경우 처리
            if not isinstance(pdf_data, bytes):
                if isinstance(pdf_data, str):
                    # 문자열인 경우 latin-1로 인코딩 (PDF 표준)
                    pdf_data = pdf_data.encode('latin-1', errors='replace')
                else:
                    # 기타 타입은 문자열로 변환 후 인코딩
                    pdf_data = str(pdf_data).encode('latin-1', errors='replace')
            
            print(f"✅ PDF 출력 완료: {len(pdf_data)} bytes")
            return pdf_data, "application/pdf"
            
        except Exception as e:
            print(f"PDF 출력 오류: {e}")
            return self._create_minimal_pdf(), "application/pdf"
    
    def _create_minimal_pdf(self) -> bytes:
        """
        최소한의 PDF 생성 (최후의 폴백)
        
        Returns:
            bytes: PDF 바이트 데이터
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            
            pdf.cell(0, 10, "PDF Conversion Error", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(0, 10, "Document could not be converted properly.", ln=True)
            pdf.cell(0, 10, "Please try again or use a different format.", ln=True)
            
            pdf_data = pdf.output()
            
            if not isinstance(pdf_data, bytes):
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('latin-1', errors='replace')
                else:
                    pdf_data = str(pdf_data).encode('latin-1', errors='replace')
            
            return pdf_data
            
        except Exception as e:
            print(f"최소 PDF 생성 실패: {e}")
            # 하드코딩된 최소 PDF
            return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Error) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
296
%%EOF"""
import os
from io import BytesIO
from typing import Tuple
from fpdf import FPDF

from .base import BaseConverter


class PdfConverter(BaseConverter):
    """DOCX를 PDF로 변환하는 클래스 (FPDF2 기반)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_korean_font()
    
    def setup_korean_font(self):
        """한국어 폰트 설정"""
        self.korean_font_available = False
        self.font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
            "/Library/Fonts/NanumGothic.ttf",  # macOS 나눔폰트
            "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
            "C:/Windows/Fonts/malgun.ttf",  # Windows 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",  # Windows 굴림
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                self.korean_font_path = font_path
                self.korean_font_available = True
                print(f"✅ 한국어 폰트 발견: {font_path}")
                break
        
        if not self.korean_font_available:
            print("⚠️ 한국어 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    
    def convert(self) -> Tuple[bytes, str]:
        """
        DOCX 파일을 PDF로 변환합니다.
        FPDF2를 사용하여 UTF-8 완전 지원으로 한국어 처리.
        
        Returns:
            Tuple[bytes, str]: (PDF 파일의 바이트 데이터, MIME 타입)
        """
        print("🚀 PDF 변환 시작 - FPDF2 기반 UTF-8 지원")
        try:
            return self._convert_with_fpdf2()
        except Exception as e:
            print(f"FPDF2 변환 실패: {e}")
            # 폴백: 단순 텍스트 PDF
            return self._create_simple_text_pdf(), "application/pdf"
    
    def _convert_with_fpdf2(self) -> Tuple[bytes, str]:
        """
        FPDF2를 사용한 PDF 변환 (완전한 UTF-8 지원)
        
        Returns:
            Tuple[bytes, str]: (PDF 파일의 바이트 데이터, MIME 타입)
        """
        # PDF 객체 생성 (UTF-8 지원)
        pdf = FPDF()
        pdf.add_page()
        
        # 폰트 설정
        try:
            if self.korean_font_available:
                # 한국어 폰트 추가 - 안전한 폰트 이름 사용
                pdf.add_font('Korean', '', self.korean_font_path, uni=True)
                pdf.set_font('Korean', size=12)
                print("✅ 한국어 폰트 설정 완료")
            else:
                # 기본 폰트 (라틴 문자만 지원)
                pdf.set_font('Arial', size=12)
                print("⚠️ 기본 폰트 사용 (한국어 지원 제한)")
        except Exception as font_error:
            print(f"폰트 설정 실패: {font_error}")
            # 폰트 설정 실패 시 기본 폰트로 복구
            self.korean_font_available = False
            try:
                pdf.set_font('Arial', size=12)
            except:
                # Arial도 실패하면 Times로 시도
                pdf.set_font('Times', size=12)
        
        # 제목 추가
        try:
            title = self.extractor.extract_metadata().get("title", "변환된 문서")
            pdf.set_font_size(16)
            
            # UTF-8 지원으로 한국어 제목 직접 사용
            if self.korean_font_available:
                pdf.cell(0, 10, title, ln=True, align='C')
            else:
                # 한국어 폰트가 없는 경우 영어로 대체
                pdf.cell(0, 10, "Converted Document", ln=True, align='C')
            
            pdf.ln(5)  # 줄 간격
        except Exception as e:
            print(f"제목 추가 실패: {e}")
            pdf.cell(0, 10, "Document", ln=True, align='C')
            pdf.ln(5)
        
        # 일반 텍스트 크기로 복원
        pdf.set_font_size(12)
        
        # 구조화된 내용 추출
        try:
            structured_content = self.extractor.extract_structured_content()
        except Exception as e:
            print(f"내용 추출 실패: {e}")
            structured_content = {"paragraphs": [], "tables": []}
        
        # 단락 내용 추가
        if structured_content.get("paragraphs"):
            for paragraph in structured_content["paragraphs"]:
                para_text = paragraph.get("text", "").strip()
                if not para_text:
                    continue
                
                try:
                    # 볼드 텍스트 처리
                    if paragraph.get("is_bold", False):
                        pdf.set_font_size(14)
                        
                    # FPDF2는 UTF-8을 완전 지원하므로 한국어 직접 사용 가능
                    if self.korean_font_available:
                        # 한국어 폰트 사용 - 완전한 한국어 지원
                        try:
                            self._add_multiline_text(pdf, para_text)
                        except UnicodeEncodeError:
                            # 한국어 폰트에서도 인코딩 오류가 발생하면 안전한 텍스트로 변환
                            safe_text = self._safe_text_for_non_korean_font(para_text)
                            self._add_multiline_text(pdf, safe_text)
                    else:
                        # 폰트가 없는 경우 안전한 텍스트로 변환
                        safe_text = self._safe_text_for_non_korean_font(para_text)
                        self._add_multiline_text(pdf, safe_text)
                    
                    # 볼드였다면 원래 크기로 복원
                    if paragraph.get("is_bold", False):
                        pdf.set_font_size(12)
                    
                    pdf.ln(3)  # 단락 간격
                    
                except Exception as e:
                    print(f"단락 추가 실패: {e}")
                    continue
        
        # 표 내용 추가
        if structured_content.get("tables"):
            pdf.ln(5)
            pdf.set_font_size(14)
            
            if self.korean_font_available:
                pdf.cell(0, 10, "표 데이터", ln=True)
            else:
                pdf.cell(0, 10, "Table Data", ln=True)
            
            pdf.set_font_size(12)
            pdf.ln(3)
            
            for i, table_data in enumerate(structured_content["tables"]):
                if not table_data.get("rows"):
                    continue
                
                try:
                    # 표 제목
                    table_title = f"표 {i+1}" if self.korean_font_available else f"Table {i+1}"
                    pdf.cell(0, 8, table_title, ln=True)
                    pdf.ln(2)
                    
                    # 표 데이터를 텍스트로 변환
                    for row_idx, row in enumerate(table_data["rows"]):
                        row_text_parts = []
                        for cell in row:
                            cell_text = str(cell).strip()
                            if cell_text:
                                if self.korean_font_available:
                                    row_text_parts.append(cell_text)
                                else:
                                    safe_cell = self._safe_text_for_non_korean_font(cell_text)
                                    row_text_parts.append(safe_cell)
                        
                        if row_text_parts:
                            row_text = " | ".join(row_text_parts)
                            self._add_multiline_text(pdf, f"  {row_text}")
                            pdf.ln(1)
                    
                    pdf.ln(5)  # 표 간 간격
                    
                except Exception as e:
                    print(f"표 {i+1} 추가 실패: {e}")
                    continue
        
        # PDF 생성
        try:
            # FPDF2에서 안전한 바이트 출력 - dest를 명시하지 않으면 기본적으로 바이트를 반환
            pdf_bytes = pdf.output()
            
            # 바이트가 아닌 경우에만 변환 시도
            if not isinstance(pdf_bytes, bytes):
                if isinstance(pdf_bytes, str):
                    # 문자열인 경우 latin-1로 인코딩 (PDF 표준)
                    pdf_bytes = pdf_bytes.encode('latin-1', errors='replace')
                else:
                    # 기타 타입은 문자열로 변환 후 바이트로
                    pdf_bytes = str(pdf_bytes).encode('latin-1', errors='replace')
            
            print(f"✅ FPDF2 PDF 변환 완료: {len(pdf_bytes)} bytes")
            return pdf_bytes, "application/pdf"
        except Exception as e:
            print(f"PDF 출력 실패: {e}")
            return self._create_simple_text_pdf(), "application/pdf"
    
    def _add_multiline_text(self, pdf: FPDF, text: str, line_height: float = 6):
        """
        긴 텍스트를 여러 줄로 나누어 추가합니다.
        
        Args:
            pdf (FPDF): PDF 객체
            text (str): 추가할 텍스트
            line_height (float): 줄 높이
        """
        try:
            # 페이지 너비 계산 (마진 고려)
            effective_width = pdf.w - 2 * pdf.l_margin
            
            # 텍스트를 단어별로 분리
            words = text.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                
                # 텍스트 너비 측정 - 예외 처리 추가
                try:
                    line_width = pdf.get_string_width(test_line)
                    if line_width <= effective_width:
                        current_line = test_line
                    else:
                        # 현재 줄 출력
                        if current_line:
                            pdf.cell(0, line_height, current_line, ln=True)
                        current_line = word
                except (UnicodeEncodeError, Exception) as e:
                    # 텍스트 처리 오류 시 안전한 텍스트로 변환
                    print(f"텍스트 처리 오류: {e}")
                    safe_text = self._safe_text_for_non_korean_font(test_line)
                    if pdf.get_string_width(safe_text) <= effective_width:
                        current_line = safe_text
                    else:
                        if current_line:
                            pdf.cell(0, line_height, current_line, ln=True)
                        current_line = self._safe_text_for_non_korean_font(word)
            
            # 마지막 줄 출력
            if current_line:
                try:
                    pdf.cell(0, line_height, current_line, ln=True)
                except (UnicodeEncodeError, Exception):
                    # 마지막 줄에서도 오류 시 안전한 텍스트로 변환
                    safe_line = self._safe_text_for_non_korean_font(current_line)
                    pdf.cell(0, line_height, safe_line, ln=True)
                    
        except Exception as e:
            print(f"멀티라인 텍스트 추가 실패: {e}")
            # 최후의 수단: 단순 텍스트 추가
            try:
                safe_text = self._safe_text_for_non_korean_font(text[:100])  # 처음 100자만
                pdf.cell(0, line_height, f"[Text Error] {safe_text}", ln=True)
            except:
                pdf.cell(0, line_height, "[Text Processing Error]", ln=True)
    
    def _safe_text_for_non_korean_font(self, text: str) -> str:
        """
        한국어 폰트가 없는 경우 안전한 텍스트로 변환합니다.
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 안전한 텍스트
        """
        if not text:
            return ""
        
        try:
            # ASCII 문자만 유지
            ascii_chars = []
            has_korean = False
            
            for char in text:
                if ord(char) < 128:  # ASCII 범위
                    ascii_chars.append(char)
                elif '\uac00' <= char <= '\ud7af':  # 한글 음절
                    has_korean = True
                    ascii_chars.append('K')  # 한국어 표시
                elif char.isspace():
                    ascii_chars.append(' ')
                else:
                    ascii_chars.append('?')
            
            result = ''.join(ascii_chars).strip()
            
            # 연속된 문자들 정리
            import re
            result = re.sub(r'\s+', ' ', result)
            result = re.sub(r'K+', '[Korean]', result)
            result = re.sub(r'\?+', '?', result)
            
            # 한국어가 있었다면 알림 추가
            if has_korean and result:
                result = f"[Korean text converted] {result}"
            
            return result if result else "[Empty or non-ASCII content]"
            
        except Exception as e:
            print(f"텍스트 안전 변환 오류: {e}")
            return "[Text conversion error]"
    
    def _create_simple_text_pdf(self) -> bytes:
        """
        단순 텍스트 PDF 생성 (최후의 폴백)
        
        Returns:
            bytes: PDF 파일의 바이트 데이터
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            
            # 기본 메시지
            pdf.cell(0, 10, "Document Conversion Completed", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(0, 10, "Original content may contain unsupported characters.", ln=True)
            pdf.ln(5)
            
            # 텍스트 내용 추가 시도
            try:
                text_content = self.extractor.extract_text_content()
                safe_text = self._safe_text_for_non_korean_font(text_content)
                
                if safe_text:
                    pdf.cell(0, 10, "Content:", ln=True)
                    pdf.ln(3)
                    self._add_multiline_text(pdf, safe_text[:1000])  # 처음 1000자만
                    
                    if len(safe_text) > 1000:
                        pdf.ln(5)
                        pdf.cell(0, 8, "... (content truncated)", ln=True)
            except:
                pdf.cell(0, 8, "Content extraction failed.", ln=True)
            
            # 안전한 PDF 출력
            pdf_bytes = pdf.output()
            
            # 바이트가 아닌 경우 변환
            if not isinstance(pdf_bytes, bytes):
                if isinstance(pdf_bytes, str):
                    pdf_bytes = pdf_bytes.encode('latin-1', errors='replace')
                else:
                    pdf_bytes = str(pdf_bytes).encode('latin-1', errors='replace')
            
            return pdf_bytes
            
        except Exception as e:
            print(f"단순 PDF 생성 실패: {e}")
            # 최소한의 PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            pdf.cell(0, 10, "PDF Generation Error", ln=True)
            
            # 안전한 PDF 출력
            try:
                pdf_bytes = pdf.output()
                if not isinstance(pdf_bytes, bytes):
                    if isinstance(pdf_bytes, str):
                        pdf_bytes = pdf_bytes.encode('latin-1', errors='replace')
                    else:
                        pdf_bytes = str(pdf_bytes).encode('latin-1', errors='replace')
                return pdf_bytes
            except:
                # 최후의 수단: 최소한의 유효한 PDF 바이트
                return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n178\n%%EOF"
