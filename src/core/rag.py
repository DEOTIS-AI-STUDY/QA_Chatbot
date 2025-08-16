"""
RAG 시스템 핵심 로직
"""
from typing import Tuple, Union
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import ElasticsearchStore
from core.config import ELASTICSEARCH_URL, INDEX_NAME
from utils.elasticsearch import ElasticsearchManager


def create_rag_chain(embeddings, llm_model, top_k: int = 3) -> Tuple[Union[RetrievalQA, None], Union[bool, str]]:
    """RAG 체인 생성"""
    try:
        # Elasticsearch 클라이언트 연결 확인
        es_client, success, message = ElasticsearchManager.get_safe_elasticsearch_client()
        if not success:
            return None, f"Elasticsearch 연결 실패: {message}"
        
        # 인덱스 존재 확인
        if not es_client.indices.exists(index=INDEX_NAME):
            return None, f"인덱스 '{INDEX_NAME}'가 존재하지 않습니다. PDF 파일을 먼저 인덱싱하세요."
        
        # 문서 개수 확인
        doc_count = es_client.count(index=INDEX_NAME).get("count", 0)
        if doc_count == 0:
            return None, f"인덱스 '{INDEX_NAME}'에 문서가 없습니다. PDF 파일을 먼저 인덱싱하세요."
        
        # Elasticsearch 벡터스토어 연결
        try:
            # 가장 기본적인 방법으로 시도
            vectorstore = ElasticsearchStore(
                embedding=embeddings,
                index_name=INDEX_NAME,
                es_url=ELASTICSEARCH_URL
            )
        except TypeError as type_error:
            # 파라미터 이름 문제인 경우
            try:
                vectorstore = ElasticsearchStore(
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                    elasticsearch_url=ELASTICSEARCH_URL
                )
            except Exception as vs_error2:
                return None, f"벡터스토어 생성 실패: {str(vs_error2)}"
        except Exception as vs_error:
            return None, f"벡터스토어 생성 실패: {str(vs_error)}"
        
        # 리트리버 설정
        try:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "fetch_k": min(top_k * 3, 10000)
                }
            )
        except Exception as ret_error:
            return None, f"리트리버 설정 실패: {str(ret_error)}"
        
        # 프롬프트 템플릿
        prompt_template = """
다음 문서 내용을 바탕으로 질문에 답변해주세요.
문서에서 답을 찾을 수 없다면 "문서에 관련 내용이 없습니다"라고 답변하세요.
답변은 친절하고 자세하게 해주세요.

문서 내용:
{context}

질문: {question}

답변:
"""
        
        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        except Exception as prompt_error:
            return None, f"프롬프트 템플릿 설정 실패: {str(prompt_error)}"
        
        # QA 체인 생성
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_model,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
        except Exception as qa_error:
            return None, f"QA 체인 생성 실패: {str(qa_error)}"
        
        # 최종 검증
        if qa_chain is None:
            return None, "QA 체인이 None으로 생성되었습니다."
        
        # 간단한 테스트 쿼리
        try:
            test_result = qa_chain({"query": "테스트"})
            if test_result is None:
                return None, "QA 체인 테스트 실패: 응답이 None입니다."
        except Exception as test_error:
            return None, f"QA 체인 테스트 실패: {str(test_error)}"
        
        return qa_chain, True
        
    except Exception as e:
        return None, f"예상치 못한 오류: {str(e)} (타입: {type(e).__name__})"
