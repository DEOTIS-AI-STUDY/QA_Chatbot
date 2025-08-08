import streamlit as st
import os
from tempfile import NamedTemporaryFile

# --- 변경/추가된 라이브러리 ---
# LLM과 임베딩 모델을 로컬로 돌리기 위한 라이브러리
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
# -----------------------------

# LangChain 관련 기존 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- OpenAI API 키 설정 부분은 이제 필요 없습니다. ---

# 1. 핵심 기능 함수 정의
def process_pdf_and_create_qa_chain(pdf_file):
    """PDF를 처리하여 QA 체인을 생성하는 함수"""
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # --- ⬇️ 1. 임베딩 모델 변경 ⬇️ ---
    # 기존 코드: embeddings = OpenAIEmbeddings()
    
    # 변경된 코드: HuggingFace에서 제공하는 한국어 특화 임베딩 모델 사용
    # 모델을 처음 실행할 때 자동으로 다운로드하며, 약 400MB 정도의 파일이 필요합니다.
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} # CPU 사용, GPU 사용 시 'cuda'로 변경
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # --- ⬆️ 임베딩 모델 변경 완료 ⬆️ ---
    
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    prompt_template = """
    주어진 문서 내용만을 사용하여 다음 질문에 답변해 주세요. 문서에서 답을 찾을 수 없다면 "문서에 관련 내용이 없습니다."라고 답변하세요.

    {context}

    질문: {question}
    답변:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # --- ⬇️ 2. LLM 모델 변경 ⬇️ ---
    # 기존 코드: llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 변경된 코드: Ollama를 통해 로컬 LLM(eeve-korean)을 사용
    llm = ChatOllama(model="llama3:8b", temperature=0)
    # --- ⬆️ LLM 모델 변경 완료 ⬆️ ---

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    os.remove(tmp_file_path)
    
    return qa_chain

# 2. Streamlit UI 구성
st.set_page_config(page_title="로컬 PDF QA 챗봇", page_icon="🤖")
st.title("📄 로컬 PDF 기반 QA 챗봇 (Ollama)")

# --- OpenAI API 키 입력 부분이 필요 없어졌습니다. ---
st.info("사이드바에서 PDF 파일을 업로드하고 'PDF 처리' 버튼을 눌러주세요.")

# 세션 상태 초기화 (동일)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바에 PDF 업로더 배치 (동일)
with st.sidebar:
    st.header("PDF 업로드")
    uploaded_file = st.file_uploader("질문할 PDF 파일을 업로드하세요.", type="pdf")
    if st.button("PDF 처리"):
        if uploaded_file is not None:
            with st.spinner("로컬 모델로 PDF를 처리 중입니다... (처음에는 모델 다운로드로 오래 걸릴 수 있습니다)"):
                st.session_state.qa_chain = process_pdf_and_create_qa_chain(uploaded_file)
                st.session_state.messages = [{"role": "assistant", "content": "PDF 처리가 완료되었습니다. 무엇이든 물어보세요!"}]
            st.success("PDF 처리가 완료되었습니다!")
        else:
            st.error("PDF 파일을 업로드해주세요.")

# 채팅 기록 표시 및 사용자 입력 처리 (동일)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.qa_chain is not None:
            with st.spinner("로컬 LLM이 답변을 생성 중입니다..."):
                response = st.session_state.qa_chain({"query": prompt})
                st.markdown(response["result"])
                with st.expander("참고 문서 보기"):
                    st.write(response["source_documents"])
        else:
            st.warning("먼저 PDF 파일을 처리해주세요.")
    
    if st.session_state.qa_chain is not None:
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
