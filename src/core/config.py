"""
통합 RAG 시스템 설정
"""
import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# ===== Elasticsearch 설정 =====
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ELASTICSEARCH_SCHEME = os.getenv("ELASTICSEARCH_SCHEME", "http")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", None)
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", None)
INDEX_NAME = os.getenv("INDEX_NAME", "unified_rag")
PDF_DIR = os.getenv("PDF_DIR", "data/pdf")

# Elasticsearch URL 자동 구성 (개별 설정이 있으면 우선 사용)
if not os.getenv("ELASTICSEARCH_URL") and os.getenv("ELASTICSEARCH_HOST"):
    auth_part = ""
    if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
        auth_part = f"{ELASTICSEARCH_USERNAME}:{ELASTICSEARCH_PASSWORD}@"
    ELASTICSEARCH_URL = f"{ELASTICSEARCH_SCHEME}://{auth_part}{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}"

# ===== BGE-M3 임베딩 모델 설정 =====
BGE_MODEL_NAME = "BAAI/bge-m3"

# ===== LLM 모델 설정 =====
LLM_MODELS = {
    "solar_10_7b": {
        "name": "Upstage SOLAR-10.7B-v1.0 (Open Source)",
        "model_id": "solar:10.7b",
        "api_key_env": None
    },
    "solar_pro_preview": {
        "name": "Upstage SOLAR-Pro Preview Instruct",
        "model_id": "upstage/solar-pro-preview-instruct",
        "api_key_env": None
    },
    "qwen2": {
        "name": "Qwen2",
        "model_id": "qwen2:7b",
        "api_key_env": None
    },
    "llama3": {
        "name": "Llama3",
        "model_id": "llama3:8b",
        "api_key_env": None
    },
    "gemma3": {
        "name": "Gemma3 12B (Ollama)",
        "model_id": "gemma3:12b",
        "api_key_env": None
    }
}

# ===== 라이브러리 가용성 확인 =====
# HuggingFace 임베딩 라이브러리
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"HuggingFace 임베딩 라이브러리 로딩 실패: {e}")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
    except ImportError as e2:
        print(f"Community HuggingFace 임베딩 라이브러리도 로딩 실패: {e2}")
        HuggingFaceEmbeddings = None
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = False

# Ollama 라이브러리
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Ollama 라이브러리 로딩 실패: {e}")
    ChatOllama = None
    OLLAMA_AVAILABLE = False

# Hugging Face Transformers 라이브러리
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from langchain_huggingface import HuggingFacePipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers 라이브러리 로딩 실패: {e}")
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        TRANSFORMERS_AVAILABLE = True
    except ImportError as e2:
        print(f"Community Transformers 라이브러리도 로딩 실패: {e2}")
        HuggingFacePipeline = None
        AutoTokenizer = None
        AutoModelForCausalLM = None
        TRANSFORMERS_AVAILABLE = False
