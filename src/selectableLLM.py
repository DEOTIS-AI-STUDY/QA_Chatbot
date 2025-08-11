import textwrap
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 1. 텍스트 파일 로드 및 문서 분할
print("1. 텍스트 파일 로드 및 문서 분할...")
print("데이터 파일 형식을 선택하세요:")
print("1. txt (data/data.txt)")
print("2. json (data/data.json)")
file_choice = input("번호 입력 (1/2): ").strip()

if file_choice == "1":
    print("data/data.txt 파일을 로드합니다.")
    loader = TextLoader(file_path='data/data.txt', encoding='utf-8')
    documents = loader.load()
    file_tag = "txt"
elif file_choice == "2":
    print("data/data.json 파일을 로드합니다.")
    import json
    from langchain_core.documents import Document
    with open('data/data.json', 'r', encoding='utf-8') as f:
        raw_json = json.load(f)
    documents = []
    for file_entry in raw_json:
        for item in file_entry["data"]:
            metadata = {
                "filename": file_entry.get("filename", ""),
                # "1depth": item.get("1depth", ""),
                # "2depth": item.get("2depth", ""),
                # "3depth": item.get("3depth", ""),
                # "4depth": item.get("4depth", ""),
                # "5depth": item.get("5depth", ""),
                # "6depth": item.get("6depth", ""),
            }
            contents = item.get("contents", "")
            if isinstance(contents, str) and contents.strip():
                doc = Document(
                    page_content=contents.strip(),
                    metadata=metadata.copy()
                )
                documents.append(doc)
    file_tag = "json"
else:
    raise ValueError("잘못된 입력입니다. 1 또는 2를 선택하세요.")

# 2. 문서 분할기(Text Splitter) 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# 3. 문서 분할 실행
split_documents = text_splitter.split_documents(documents)

# --- 임베딩 모델 선택 ---
huggingface_model_name = "jhgan/ko-sroberta-multitask"    # <- 원하는 모델명으로 수정
ollama_model_name = "mxbai-embed-large"                   # <- 원하는 모델명으로 수정
openai_model_name = "text-embedding-3-small"              # <- 원하는 모델명으로 수정

print("임베딩 모델을 선택하세요:")
print(f"1. HuggingFace  (모델명: {huggingface_model_name})")
print(f"2. Ollama       (모델명: {ollama_model_name})")
print(f"3. OpenAI       (모델명: {openai_model_name})")
embedding_choice = input("번호 입력 (1/2/3): ").strip()

if embedding_choice == "1":
    print(f"HuggingFace 임베딩 모델을 사용합니다. (모델명: {huggingface_model_name})")
    embedding_function = HuggingFaceEmbeddings(model_name=huggingface_model_name)
    embed_tag = "hf"
elif embedding_choice == "2":
    print(f"Ollama 임베딩 모델을 사용합니다. (모델명: {ollama_model_name})")
    embedding_function = OllamaEmbeddings(model=ollama_model_name)
    embed_tag = "ollama"
elif embedding_choice == "3":
    print(f"OpenAI 임베딩 모델을 사용합니다. (모델명: {openai_model_name})")
    embedding_function = OpenAIEmbeddings(model=openai_model_name, api_key=os.environ.get("OPENAI_API_KEY"))
    embed_tag = "openai"
else:
    raise ValueError("잘못된 입력입니다. 1, 2, 3 중 하나를 선택하세요.")

# --- 벡터 DB 파일명 결정 ---
db_dir = "db"
os.makedirs(db_dir, exist_ok=True)
db_file = f"{db_dir}/chroma_{file_tag}_{embed_tag}"

# 5. Chroma DB 생성 또는 로드
print(f"3. Chroma DB 생성 또는 로드... (저장 위치: {db_file})")
if os.path.exists(db_file):
    print("기존 벡터 DB가 존재합니다. 해당 DB를 사용합니다.")
    vectorstore = Chroma(persist_directory=db_file, embedding_function=embedding_function)
else:
    print("벡터 DB가 존재하지 않아 새로 생성합니다.")
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_function,
        persist_directory=db_file
    )

# 6. RAG를 위한 Retriever 설정
print("4. RAG를 위한 Retriever 설정...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 7. LLM 모델 선택 및 로드
huggingface_llm_model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
ollama_llm_model_name = "llama3.2:1b" 
openai_llm_model_name = "gpt-4o-mini"

print("LLM(생성형 언어모델)을 선택하세요:")
print(f"1. HuggingFace 모델 (모델명: {huggingface_llm_model_name})")
print(f"2. Ollama 모델      (모델명: {ollama_llm_model_name})")
print(f"3. OpenAI 모델      (모델명: {openai_llm_model_name})")
llm_choice = input("번호 입력 (1/2/3): ").strip()

if llm_choice == "1":
    print(f"HuggingFace LLM 모델을 사용합니다. (모델명: {huggingface_llm_model_name})")
    from transformers import pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=huggingface_llm_model_name,
        # device="cuda"  # GPU 사용 시 활성화
    )

    from langchain_huggingface import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipe)
elif llm_choice == "2":
    print(f"Ollama LLM 모델을 사용합니다. (모델명: {ollama_llm_model_name})")
    llm = OllamaLLM(model=ollama_llm_model_name, temperature=0.0)
elif llm_choice == "3":
    print(f"OpenAI LLM 모델을 사용합니다. (모델명: {openai_llm_model_name})")
    llm = ChatOpenAI(model=openai_llm_model_name, temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY"))
else:
    raise ValueError("잘못된 입력입니다. 1, 2, 3 중 하나를 선택하세요.")

# 8. RAG 체인 구축 (쿼리 변환 기능 포함)
print("6. RAG 체인 구축 ...")

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

template = """
당신은 주어진 문맥(context)을 바탕으로 질문(question)에 답변하는 유용한 도우미입니다.
문맥을 참고하여 질문에 자세히 친절하게 답변해주세요.
문맥을 참고하여 답변을 ~다 처럼 딱딱한 형식 보다는 ~요, ~해요 등으로 부드럽게 작성해주세요.
만약 문맥에 답변할 내용이 없다면, 반드시 "답변 드릴 내용이 없어요."라고만 말해주세요.
모든 답변은 마크다운 형식으로 작성해주세요.

<context>
{context}
</context>

질문: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("\n7. 질문 및 답변을 생성합니다. (종료하려면 'exit' 입력)")
while True:
    question = input("질문 입력: ")
    if question.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break

    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)

    with open("retrieved_docs.log", "w", encoding="utf-8") as f:
        # 프롬프트 기록
        f.write("--- LLM 프롬프트 ---\n")
        f.write(template.format(context=formatted_context, question=question))
        f.write("\n\n")
        # 벡터 DB 결과 기록
        for i, doc in enumerate(retrieved_docs, 1):
            f.write(f"--- Document {i} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")

    response = rag_chain.invoke(question)

    # 응답 처리: str 타입이면 그대로, 아니면 .content 사용
    if isinstance(response, str):
        wrapped_response = textwrap.fill(response, width=80)
    else:
        wrapped_response = textwrap.fill(response.content, width=80)
    print("=*"*40)
    print(f"답변: \n{wrapped_response}\n")
    print("=*"*40)