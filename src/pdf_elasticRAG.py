#!/usr/bin/python3.11


"""
[TEXT-ONLY CORE] PDF RAG
- 지정 폴더의 PDF 읽기
- Elasticsearch + 벡터 임베딩 색인

- python pdf_elasticRAG.py 로 실행 시, 임베딩 - 인덱싱.
- python pdf_elasticRAG.py --q "질의내용" 으로 질의하면 RAG 응답
"""

# ===== Imports =====
import os
import sys
import argparse
from dotenv import load_dotenv  # [변경] .env 로드

from elasticsearch import Elasticsearch
from langchain_community.document_loaders import PyPDFLoader  # [TEXT-ONLY]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import ElasticVectorSearch

# 실행 모드
MODE_GOOGLE = False

# # ===== .env 로드 =====
load_dotenv()
if(MODE_GOOGLE):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY:
        print("⚠️ GOOGLE_API_KEY를 .env 파일 또는 환경변수로 설정하세요.")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY를 .env 파일 또는 환경변수로 설정하세요.")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY and not GOOGLE_API_KEY:
    print("⚠️ 설정된 AI API KEY 가 없습니다. (GOOGLE_API_KEY / OPENAI_API_KEY) 를 (.env/환경변수) 로 설정하세요.")
    print("종료합니다....")
    exit(-1)

# ===== Config =====
PDF_DIR = os.getenv("PDF_DIR", "pdf")  # 읽을 PDF 폴더
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "pdf_rag")

# ===== CLI =====
parser = argparse.ArgumentParser(description="PDF RAG (텍스트만 → ES 벡터 색인/질의)")
parser.add_argument("--q", dest="query", type=str, help="질의 텍스트")
parser.add_argument("--k", dest="top_k", type=int, default=9999, help="검색 상위 k (기본 8)")
args = parser.parse_args()

query_mode = args.query is not None
query_text = args.query
top_k = args.top_k


# ===== Helpers =====
def list_pdfs(pdf_dir: str):
    """폴더 내 .pdf / .PDF 전부 수집"""
    try:
        names = os.listdir(pdf_dir)
    except FileNotFoundError:
        return []
    files = []
    for name in names:
        if name.lower().endswith(".pdf"):
            p = os.path.join(pdf_dir, name)
            if os.path.isfile(p):
                files.append(os.path.abspath(p))
    return sorted(files)

# def list_pdfs(pdf_dir: str):
#     try:
#         names = os.listdir(pdf_dir)
#     except FileNotFoundError:
#         return []
#     return sorted(
#         os.path.abspath(os.path.join(pdf_dir, n))
#         for n in names
#         if n.lower().endswith(".pdf") and os.path.isfile(os.path.join(pdf_dir, n))
#     )

def total_text_len(docs):
    return sum(len((d.page_content or "").strip()) for d in docs)


        


# def print_sources(docs):
    # print("\n🔎 Top-k 문서 출처:")
    # for i, d in enumerate(docs, 1):
    #     print(f"- #{i}: source={d.metadata.get('source')}, page={d.metadata.get('page_number')}")


# ===== Indexing =====
if not query_mode:
    es = Elasticsearch(ELASTICSEARCH_URL)

    # 인덱스 초기화(깨끗하게 시작)
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    pdf_files = list_pdfs(PDF_DIR)
    print("📄 대상 파일:", pdf_files)
    if not pdf_files:
        print(f"❌ PDF 디렉토리에 파일이 없습니다: {PDF_DIR}")
        sys.exit(1)

    all_documents = []
    for pdf_path in pdf_files:
        print("file :", pdf_path)

        p = PyPDFLoader(pdf_path)
        docs = p.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        docs = chunks
        if total_text_len(docs) > 0:
            # 메타 최소 보강
            for d in docs:
                md = d.metadata or {}
                md["category"] = md.get("category", "Text")
                d.metadata = md

        # 메타데이터에 source/filename 보강
        # print("####### extracted by ",loader_used," #######")
        for d in docs:
            # print(d.page_content)
            md = d.metadata or {}
            md["source"] = pdf_path
            md["filename"] = os.path.basename(pdf_path)
            d.metadata = md
        all_documents.extend(docs)
        print(f"docs={len(docs)}, text_len={total_text_len(docs)}")

    if not all_documents:
        print("❌ 추출된 텍스트가 없습니다. (스캔 PDF일 수 있음)")
        sys.exit(1)

    if(MODE_GOOGLE):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    ElasticVectorSearch.from_documents(
        all_documents,
        embedding=embeddings,
        elasticsearch_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
    )

    es.indices.refresh(index=INDEX_NAME)
    cnt = es.count(index=INDEX_NAME).get("count", 0)
    print(f"✅ 색인 완료. ES 문서 수: {cnt}")
    print('➡️ 질의: python script.py --q "질문" --k 8')


# ===== Querying =====
else:
    if(MODE_GOOGLE):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = ElasticVectorSearch(
        embedding=embeddings,
        elasticsearch_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query_text)
    # print_sources(docs)

    if(MODE_GOOGLE):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""[문서]
{context}

[질문]
{query_text}

[지침]
- 당신은 상담원입니다. 친절한 말투로 답하세요.
- 문서에 근거해 한국어로 답하세요.
- 문서에 정보가 없거나 애매하면, 문서 내에 해당 정보가 없음을 명시하고 필요한 추가 질문을 하세요.
- 마지막에 사용한 근거를 한 줄로 요약과 실제 문서 내용 일부를 마지막에 제시하세요.

[답변]"""

    resp = llm.invoke(prompt)
    if(MODE_GOOGLE):
        print("\n🤖 Gemini 응답:")
    else:
        print("\n🤖 GPT 응답:")
    print(resp.content)
