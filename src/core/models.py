"""
임베딩 및 LLM 모델 팩토리
"""
import os
from typing import Optional
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
        
        if TRANSFORMERS_AVAILABLE:
            available_models["solar_pro_preview"] = LLM_MODELS["solar_pro_preview"]
            
        return available_models
