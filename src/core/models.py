"""
임베딩 및 LLM 모델 팩토리
"""
import os
from typing import Optional, List, Dict, Any, Tuple
from core.config import (
    BGE_MODEL_NAME, LLM_MODELS,
    HUGGINGFACE_EMBEDDINGS_AVAILABLE, OLLAMA_AVAILABLE, TRANSFORMERS_AVAILABLE,
    HuggingFaceEmbeddings, ChatOllama
)

# TRANSFORMERS_AVAILABLE이 True일 때만 import
if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except ImportError:
            TRANSFORMERS_AVAILABLE = False


class ModelFactory:
    @staticmethod
    def add_gemma3_model():
        """LLM_MODELS에 gemma3:12b 모델 추가 (Ollama 기반)"""
        if "gemma3" not in LLM_MODELS:
            LLM_MODELS["gemma3"] = {
                "name": "Gemma3 12B (Ollama)",
                "model_id": "gemma3:12b",
                "api_key_env": None
            }
    """모델 생성 팩토리 클래스"""
    
    @staticmethod
    def create_embedding_model() -> Optional[HuggingFaceEmbeddings]:
        """BGE-M3 임베딩 모델 생성"""
        if not HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            print("HuggingFace 임베딩 라이브러리를 사용할 수 없습니다.")
            return None
            
        try:
            return HuggingFaceEmbeddings(
                model_name=BGE_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"BGE-M3 임베딩 모델 로딩 실패: {str(e)}")
            # 폴백으로 기본 한국어 모델 사용
            try:
                return HuggingFaceEmbeddings(
                    model_name="jhgan/ko-sroberta-multitask",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e2:
                print(f"폴백 임베딩 모델도 로딩 실패: {str(e2)}")
                return None
    
    @staticmethod
    def create_llm_model(model_choice: str):
        """선택된 LLM 모델 생성 (Upstage API Key 없이 HuggingFace Transformers만 사용)"""
        # Gemma3:12b (Ollama 기반)
        if model_choice == "gemma3":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다. pip install langchain-ollama"
            try:
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["gemma3"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ Gemma3 12B (Ollama) 모델 생성 성공"
            except Exception as e:
                return None, f"Gemma3 12B (Ollama) 모델 생성 실패: {str(e)}"
        
        # Gemma3:27b (Ollama 기반)
        elif model_choice == "gemma3_big":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다. pip install langchain-ollama"
            try:
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["gemma3_big"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ Gemma3 27B (Ollama) 모델 생성 성공"
            except Exception as e:
                return None, f"Gemma3 27B (Ollama) 모델 생성 실패: {str(e)}"
        
        # Gemma3:27b-it-qat (Ollama 기반)
        elif model_choice == "gemma3_big_qat":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다. pip install langchain-ollama"
            try:
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["gemma3_big_qat"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ Gemma3 27B IT QAT (Ollama) 모델 생성 성공"
            except Exception as e:
                return None, f"Gemma3 27B IT QAT (Ollama) 모델 생성 실패: {str(e)}"
        if model_choice == "upstage":
            if not TRANSFORMERS_AVAILABLE:
                return None, "❌ Transformers 라이브러리가 설치되지 않았습니다. pip install transformers torch"
            try:
                model_id = LLM_MODELS["upstage"]["model_id"]
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    return_full_text=False
                )
                llm_model = HuggingFacePipeline(pipeline=text_pipeline)
                return llm_model, "✅ Upstage Solar Mini (로컬) 모델 생성 성공"
            except Exception as e:
                return None, f"Upstage Solar Mini (로컬) 모델 생성 실패: {str(e)}"

        elif model_choice == "solar_10_7b":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다. pip install langchain-ollama"
            
            try:
                # 환경 변수에서 Ollama URL 가져오기
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["solar_10_7b"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ SOLAR-10.7B 오픈소스 모델 생성 성공"
            except Exception as e:
                return None, f"SOLAR-10.7B 모델 생성 실패: {str(e)}"
        
        elif model_choice == "qwen2":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다."
            
            try:
                # 환경 변수에서 Ollama URL 가져오기
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["qwen2"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ Qwen2 모델 생성 성공"
            except Exception as e:
                return None, f"Qwen2 모델 생성 실패: {str(e)}"
        
        elif model_choice == "solar_pro_preview":
            if not TRANSFORMERS_AVAILABLE:
                return None, "❌ Transformers 라이브러리가 설치되지 않았습니다. pip install transformers torch"
            try:
                model_id = LLM_MODELS["solar_pro_preview"]["model_id"]
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                    offload_folder="offload",
                    offload_buffers=True
                )
                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    return_full_text=False
                )
                llm_model = HuggingFacePipeline(pipeline=text_pipeline)
                return llm_model, "✅ SOLAR-Pro Preview (로컬) 모델 생성 성공"
            except Exception as e:
                return None, f"SOLAR-Pro Preview (로컬) 모델 생성 실패: {str(e)}"
        
        elif model_choice == "llama3":
            if not OLLAMA_AVAILABLE:
                return None, "❌ Ollama 라이브러리가 설치되지 않았습니다."
            
            try:
                # 환경 변수에서 Ollama URL 가져오기
                ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = ChatOllama(
                    model=LLM_MODELS["llama3"]["model_id"],
                    temperature=0,
                    base_url=ollama_base_url
                )
                return model, "✅ Llama3 모델 생성 성공"
            except Exception as e:
                return None, f"Llama3 모델 생성 실패: {str(e)}"
        
        else:
            return None, f"지원하지 않는 모델: {model_choice}"
    
    @staticmethod
    def get_available_models() -> dict:
        """사용 가능한 모델 목록 반환"""
        available_models = {}
        if OLLAMA_AVAILABLE:
            available_models["qwen2"] = LLM_MODELS["qwen2"]
            available_models["llama3"] = LLM_MODELS["llama3"]
            available_models["solar_10_7b"] = LLM_MODELS["solar_10_7b"]
            if "gemma3" in LLM_MODELS:
                available_models["gemma3"] = LLM_MODELS["gemma3"]
            if "gemma3_big" in LLM_MODELS:
                available_models["gemma3_big"] = LLM_MODELS["gemma3_big"]
            if "gemma3_big_qat" in LLM_MODELS:
                available_models["gemma3_big_qat"] = LLM_MODELS["gemma3_big_qat"]
        if TRANSFORMERS_AVAILABLE:
            available_models["solar_pro_preview"] = LLM_MODELS["solar_pro_preview"]
        return available_models

    # =============================================================================
    # Ollama 모델 관리 메서드들
    # =============================================================================
    
    @staticmethod
    def get_ollama_manager():
        """OllamaManager 인스턴스 반환"""
        try:
            from utils.ollama_manager import OllamaManager
            return OllamaManager()
        except ImportError as e:
            print(f"OllamaManager import 실패: {e}")
            return None
    
    @staticmethod
    def list_ollama_models() -> Tuple[List[Dict[str, Any]], str]:
        """설치된 Ollama 모델 목록 조회"""
        if not OLLAMA_AVAILABLE:
            return [], "Ollama가 사용 불가능합니다"
        
        ollama_manager = ModelFactory.get_ollama_manager()
        if not ollama_manager:
            return [], "OllamaManager를 초기화할 수 없습니다"
        
        # Ollama 연결 확인
        connected, message = ollama_manager.check_ollama_connection()
        if not connected:
            return [], f"Ollama 연결 실패: {message}"
        
        return ollama_manager.list_models()
    
    @staticmethod
    def add_ollama_model(model_name: str) -> Tuple[bool, str]:
        """Ollama 모델 추가 (다운로드)"""
        if not OLLAMA_AVAILABLE:
            return False, "Ollama가 사용 불가능합니다"
        
        ollama_manager = ModelFactory.get_ollama_manager()
        if not ollama_manager:
            return False, "OllamaManager를 초기화할 수 없습니다"
        
        # Ollama 연결 확인
        connected, message = ollama_manager.check_ollama_connection()
        if not connected:
            return False, f"Ollama 연결 실패: {message}"
        
        # 모델 다운로드
        success, result_message = ollama_manager.pull_model(model_name)
        
        if success:
            # LLM_MODELS에 자동으로 추가
            ModelFactory.register_ollama_model_to_config(model_name)
        
        return success, result_message
    
    @staticmethod 
    def remove_ollama_model(model_name: str) -> Tuple[bool, str]:
        """Ollama 모델 삭제"""
        if not OLLAMA_AVAILABLE:
            return False, "Ollama가 사용 불가능합니다"
        
        ollama_manager = ModelFactory.get_ollama_manager()
        if not ollama_manager:
            return False, "OllamaManager를 초기화할 수 없습니다"
        
        # Ollama 연결 확인
        connected, message = ollama_manager.check_ollama_connection()
        if not connected:
            return False, f"Ollama 연결 실패: {message}"
        
        # 모델 삭제
        success, result_message = ollama_manager.delete_model(model_name)
        
        if success:
            # LLM_MODELS에서도 제거
            ModelFactory.unregister_ollama_model_from_config(model_name)
        
        return success, result_message
    
    @staticmethod
    def register_ollama_model_to_config(model_name: str):
        """LLM_MODELS 설정에 Ollama 모델 등록"""
        # 안전한 키 생성 (특수문자 제거)
        safe_key = model_name.replace(":", "_").replace("-", "_").replace(".", "_")
        
        if safe_key not in LLM_MODELS:
            LLM_MODELS[safe_key] = {
                "name": f"{model_name} (Ollama)",
                "model_id": model_name,
                "api_key_env": None
            }
            print(f"✅ 모델 '{model_name}' 을 설정에 등록했습니다 (키: {safe_key})")
    
    @staticmethod
    def unregister_ollama_model_from_config(model_name: str):
        """LLM_MODELS 설정에서 Ollama 모델 제거"""
        # 안전한 키 생성
        safe_key = model_name.replace(":", "_").replace("-", "_").replace(".", "_")
        
        if safe_key in LLM_MODELS:
            del LLM_MODELS[safe_key]
            print(f"✅ 모델 '{model_name}' 을 설정에서 제거했습니다 (키: {safe_key})")
    
    @staticmethod
    def get_available_ollama_models_from_library() -> List[str]:
        """Ollama 라이브러리에서 다운로드 가능한 모델 목록"""
        ollama_manager = ModelFactory.get_ollama_manager()
        if not ollama_manager:
            return []
        
        return ollama_manager.get_available_models_from_library()
    
    @staticmethod
    def check_ollama_status() -> Tuple[bool, str, Dict[str, Any]]:
        """Ollama 서버 상태 및 설치된 모델 정보 확인"""
        if not OLLAMA_AVAILABLE:
            return False, "Ollama가 사용 불가능합니다", {}
        
        ollama_manager = ModelFactory.get_ollama_manager()
        if not ollama_manager:
            return False, "OllamaManager를 초기화할 수 없습니다", {}
        
        # 연결 확인
        connected, message = ollama_manager.check_ollama_connection()
        if not connected:
            return False, message, {}
        
        # 모델 목록 조회
        models, list_message = ollama_manager.list_models()
        
        status_info = {
            "server_url": ollama_manager.base_url,
            "models_count": len(models),
            "models": models,
            "list_message": list_message
        }
        
        return True, "Ollama 서버 정상 작동", status_info
