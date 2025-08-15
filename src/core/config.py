"""
통합 RAG 시스템 설정
"""
import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# ===== Elasticsearch 설정 =====
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "unified_rag")
PDF_DIR = os.getenv("PDF_DIR", "pdf")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ===== BGE-M3 임베딩 모델 설정 =====
BGE_MODEL_NAME = "BAAI/bge-m3"

# ===== LLM 모델 설정 =====
LLM_MODELS = {
    "upstage": {
        "name": "Upstage Solar LLM",
        "model_id": "solar-1-mini-chat",
        "api_key_env": "UPSTAGE_API_KEY"
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
    }
}

# ===== 라이브러리 가용성 확인 =====
# Langsmith 라이브러리
try:
    from langsmith import Client
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

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

# Upstage 라이브러리
try:
    from langchain_upstage import ChatUpstage
    UPSTAGE_AVAILABLE = True
except ImportError as e:
    print(f"Upstage 라이브러리 로딩 실패: {e}")
    ChatUpstage = None
    UPSTAGE_AVAILABLE = False

# Ollama 라이브러리
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError as e:
    print(f"Ollama 라이브러리 로딩 실패: {e}")
    ChatOllama = None
    OLLAMA_AVAILABLE = False
