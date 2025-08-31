"""
Ollama 모델 관리 유틸리티
"""
import os
import requests
import json
from typing import List, Dict, Any, Tuple, Optional
from core.config import OLLAMA_AVAILABLE


class OllamaManager:
    """Ollama 모델 관리 클래스"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.api_base = f"{self.base_url}/api"
    
    def check_ollama_connection(self) -> Tuple[bool, str]:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return True, "Ollama 서버 연결 성공"
            else:
                return False, f"Ollama 서버 응답 오류: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
        except Exception as e:
            return False, f"Ollama 연결 확인 중 오류: {str(e)}"
    
    def list_models(self) -> Tuple[List[Dict[str, Any]], str]:
        """설치된 Ollama 모델 목록 조회"""
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                # 모델 정보를 정리하여 반환
                model_list = []
                for model in models:
                    model_info = {
                        'name': model.get('name', ''),
                        'size': model.get('size', 0),
                        'digest': model.get('digest', ''),
                        'modified_at': model.get('modified_at', ''),
                        'details': model.get('details', {})
                    }
                    model_list.append(model_info)
                
                return model_list, "모델 목록 조회 성공"
            else:
                return [], f"모델 목록 조회 실패: {response.status_code}"
        except Exception as e:
            return [], f"모델 목록 조회 중 오류: {str(e)}"
    
    def pull_model(self, model_name: str) -> Tuple[bool, str]:
        """Ollama 모델 다운로드/설치"""
        try:
            # 스트리밍 요청으로 다운로드 진행상황 확인
            payload = {"name": model_name}
            response = requests.post(
                f"{self.api_base}/pull",
                json=payload,
                stream=True,
                timeout=300  # 5분 타임아웃
            )
            
            if response.status_code == 200:
                # 스트리밍 응답 처리
                progress_info = []
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            status = data.get('status', '')
                            if status:
                                progress_info.append(status)
                                print(f"📥 {model_name}: {status}")
                            
                            # 완료 확인
                            if data.get('status') == 'success' or 'success' in status.lower():
                                return True, f"모델 '{model_name}' 다운로드 완료"
                        except json.JSONDecodeError:
                            continue
                
                return True, f"모델 '{model_name}' 다운로드 완료"
            else:
                return False, f"모델 다운로드 실패: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return False, f"모델 '{model_name}' 다운로드 시간 초과 (5분)"
        except Exception as e:
            return False, f"모델 다운로드 중 오류: {str(e)}"
    
    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """Ollama 모델 삭제"""
        try:
            payload = {"name": model_name}
            response = requests.delete(
                f"{self.api_base}/delete",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return True, f"모델 '{model_name}' 삭제 완료"
            else:
                return False, f"모델 삭제 실패: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"모델 삭제 중 오류: {str(e)}"
    
    def show_model_info(self, model_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """특정 모델의 상세 정보 조회"""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.api_base}/show",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json(), "모델 정보 조회 성공"
            else:
                return None, f"모델 정보 조회 실패: {response.status_code}"
                
        except Exception as e:
            return None, f"모델 정보 조회 중 오류: {str(e)}"
    
    def get_available_models_from_library(self) -> List[str]:
        """Ollama 라이브러리에서 사용 가능한 모델 목록 (일반적인 모델들)"""
        return [
            "llama3.1:8b",
            "llama3.1:70b", 
            "llama3.2:3b",
            "llama3.2:1b",
            "gemma2:9b",
            "gemma2:27b",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "qwen2.5:32b",
            "mistral:7b",
            "mixtral:8x7b",
            "codellama:7b",
            "codellama:13b",
            "phi3:3.8b",
            "phi3:14b",
            "yi:6b",
            "yi:34b",
            "deepseek-coder:6.7b",
            "deepseek-coder:33b",
            "neural-chat:7b",
            "openchat:7b",
            "starling-lm:7b",
            "vicuna:7b",
            "vicuna:13b"
        ]
    
    def format_size(self, size_bytes: int) -> str:
        """바이트를 읽기 쉬운 형태로 변환"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
