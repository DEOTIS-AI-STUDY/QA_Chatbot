# 명령어 모음

```bash

# 애플리케이션 실행
lsof -i :8110 | grep LISTEN | awk '{print $2}' | xargs kill -9 && sleep 2 && cd /@@@@/deotis_qa_chatbot && .venv/bin/python src/api/main.py

# 가상환경
source /@@@/deotis_qa_chatbot/.venv/bin/activate

-------

# 인덱싱

## 기존 데이터 유지하며 파일 타입만 새롭게 추가
python src/api/main.py --init-index --file-types json
python src/api/main.py --init-index --file-types pdf
python src/api/main.py --init-index --file-types txt
python src/api/main.py --init-index --file-types docx
python src/api/main.py --init-index --file-types all

## 새로운 --only 옵션 (기존 데이터 삭제 후 json만 인덱싱)
python src/api/main.py --init-index --file-types json --only

## 다른 파일 타입들도 동일하게 사용 가능
python src/api/main.py --init-index --file-types pdf --only
python src/api/main.py --init-index --file-types txt --only
python src/api/main.py --init-index --file-types docx --only
python src/api/main.py --init-index --file-types all --only

-------

# docx -> json

## 기본 변환
python src/preprocess/run_docx_to_index.py data/docx

## 분석과 비교를 포함한 변환
python src/preprocess/run_docx_to_index.py data/docx --compare data/index.json --analyze

## 특정 출력 파일로 저장
python src/preprocess/run_docx_to_index.py data/docx -o custom_output.json

-------

# elasticsearch

## 'my-index' 라는 이름의 인덱스를 삭제
curl -X DELETE "http://localhost:9200/unified_rag" -v

## 인덱스 삭제 확인
curl -s http://localhost:9200/_cat/indices?v

## docker 재기동
docker-compose down && docker-compose up -d
```

<br/>

# QA 답변 정확도 영향 요소

## 🎯 RAG QA 답변 정확도 영향 요소

### 1. **데이터 전처리 단계**

#### **1.1 파일 타입별 처리 품질**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)

# PDF 처리
- 표 추출 정확도: pdfplumber vs PyPDF의 차이
- 텍스트 인식 오류: OCR 품질, 스캔 문서 처리
- 페이지 구조 보존: 헤더, 푸터, 레이아웃 정보

# JSON 처리
- 구조화 정보 활용: title, heading, section 메타데이터
- 키워드 추출 품질: _extract_keywords() 함수의 패턴 매칭
- 카테고리 분류 정확도: _extract_category() 로직
```

```python
# 📂 src/utils/elasticsearch_index.py
class ElasticsearchIndexer:

    def _enhanced_pdf_loader(self, file_path: str) -> List[Document]:
        """PDF 표 추출 및 인라인 통합"""
        # PDF 처리 품질이 검색 정확도에 직접 영향

    def _process_json_file(self, file_path: str) -> List[Document]:
        """JSON 파일을 구조화된 문서로 변환"""
        # 메타데이터 추출 품질이 검색 성능 좌우

    def _extract_keywords(self, content: str) -> str:
        """키워드 자동 추출"""
        # 키워드 패턴의 완성도가 검색 정확도에 큰 영향

    def _extract_category(self, title: str, heading: str) -> str:
        """카테고리 자동 분류"""
        # 잘못된 분류 시 필터링 검색에서 누락 발생

    def _classify_content_type(self, content: str) -> str:
        """콘텐츠 타입 분류"""
        # 문서 유형 분류 정확도가 검색 품질 영향
```

```python
# 📂 src/utils/pdf_table_extractor.py
class PDFTableExtractor:

    def extract_tables_and_text(self, pdf_path: str) -> List[Document]:
        """PDF에서 표와 텍스트를 통합 추출"""
        # 표 추출 정확도가 답변 완성도에 직접 영향

    def table_to_markdown(self, table_data) -> str:
        """표 데이터를 마크다운으로 변환"""
        # 표 구조 보존 품질이 답변 정확도 좌우
```

#### **1.2 표 데이터 처리**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)
# 📂 src/utils/elasticsearch_index.py
def _is_table_content(self, content: str, metadata: dict = None) -> bool:
    """표 감지 정확도가 답변 품질에 직접 영향"""

    # 잘못된 표 감지 → 청킹으로 표 분할 → 불완전한 정보
    # 정확한 표 감지 → 완전한 표 보존 → 정확한 답변

    # 현재 감지 로직의 한계점:
    - 마크다운 패턴 의존성 (50% 임계값)
    - 복잡한 표 구조 미인식
    - False Positive/Negative 발생 가능
```

### 2. **텍스트 청킹 전략**

#### **2.1 청킹 파라미터**

```python
# 영향도: ⭐⭐⭐⭐ (높음)
# 📂 src/utils/elasticsearch_index.py
@staticmethod
def get_optimized_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,        # 너무 작으면: 문맥 손실
        chunk_overlap=160,     # 너무 크면: 중복 증가, 속도 저하
        length_function=len,   # 토큰 vs 문자 단위 차이
        separators=["\n\n", "\n", " ", ""]  # 분할 우선순위
    )

# 정확도 영향:
- chunk_size가 작을수록: 정밀하지만 문맥 부족
- chunk_size가 클수록: 문맥 풍부하지만 노이즈 증가
- overlap이 적절하지 않으면: 정보 유실 또는 중복
```

#### **2.2 지능적 청킹 로직**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)
# 📂 src/utils/elasticsearch_index.py
def _index_files_by_type(self, file_type: str, files: List[str], embeddings, hybrid_tracker):
    """청킹 결정 로직이 정보 완성도에 결정적 영향"""

    for doc in docs:
        if self._is_table_content(doc.page_content, doc.metadata):
            # 표는 분할하지 않음 → 완전한 정보 보존
            all_documents.append(doc)
        else:
            # 일반 텍스트는 청킹 → 적절한 크기로 분할
            chunks = splitter.split_documents([doc])
            all_documents.extend(chunks)

# 문제점:
- 표 감지 실패 시 중요 정보 분할
- 긴 절차 설명이 여러 청크로 분리
- 관련 정보가 다른 청크에 분산
```

### 3. **임베딩 모델 선택**

#### **3.1 임베딩 모델 성능**

```python
# 영향도: ⭐⭐⭐⭐ (높음)

# 선택 기준:
- 한국어 금융 도메인 특화 성능
- 벡터 차원수 (1536 vs 다른 크기)
- 컨텍스트 길이 제한
```

### 4. **검색 시스템 품질**

#### **4.1 하이브리드 검색 가중치**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)

search_query = {
    "should": [
        {"match": {"metadata.title": {"query": query, "boost": 3.0}}},      # 제목 가중치
        {"match": {"metadata.heading": {"query": query, "boost": 2.0}}},    # 헤딩 가중치
        {"match": {"metadata.keywords": {"query": query, "boost": 2.5}}},   # 키워드 가중치
        {"match": {"text": {"query": query, "boost": 1.0}}}                # 본문 가중치
    ]
}

# 📂 src/utils/elasticsearch.py
class ElasticsearchManager:

    def keyword_search(self, query: str, top_k: int = 5):
        """키워드 검색 가중치 설정이 검색 품질 결정"""

        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"metadata.title": {"query": query, "boost": 3.0}}},      # 제목 가중치
                        {"match": {"metadata.heading": {"query": query, "boost": 2.0}}},    # 헤딩 가중치
                        {"match": {"metadata.keywords": {"query": query, "boost": 2.5}}},   # 키워드 가중치
                        {"match": {"text": {"query": query, "boost": 1.0}}}                # 본문 가중치
                    ]
                }
            }
        }

        # 최적화 포인트:
        # - 각 필드별 가중치 조정
        # - 도메인 특성에 맞는 가중치 설정
        # - A/B 테스트를 통한 최적값 찾기

    def semantic_search(self, query: str, top_k: int = 5):
        """벡터 검색 파라미터가 의미적 검색 품질 좌우"""

        search_query = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 50,  # 후보군 확장이 정확도 향상
            }
        }

    def hybrid_search(self, query: str, top_k: int = 5):
        """하이브리드 검색 융합이 최종 검색 품질 결정"""
        return self._combine_search_results(keyword_results, semantic_results)
```

# 최적화 포인트:

- 각 필드별 가중치 조정
- 도메인 특성에 맞는 가중치 설정
- A/B 테스트를 통한 최적값 찾기

````

#### **4.2 RRF (Reciprocal Rank Fusion) 파라미터**

```python
# 영향도: ⭐⭐⭐⭐ (높음)
# 📂 src/utils/elasticsearch.py
def _combine_search_results(self, keyword_results, semantic_results, k=60):
    """RRF 파라미터가 검색 결과 품질에 직접 영향"""

    # k값이 검색 품질에 미치는 영향:
    # k가 작을수록 (k=20): 상위 순위 결과에 높은 가중치
    # k가 클수록 (k=100): 순위 간 점수 차이 감소

    rrf_score = 1.0 / (k + rank)

    combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

# ElasticsearchStore 설정:
# 📂 src/core/rag.py
def create_rag_chain(embeddings, llm_model, top_k: int = 3):
    """ElasticsearchStore RRF 설정이 검색 성능 좌우"""

    vectorstore = ElasticsearchStore(
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(
            hybrid=True,
            rrf={
                "window_size": 100,    # 후보 문서 수 (많을수록 정확하지만 느림)
                "rank_constant": 20    # RRF k값
            }
        )
    )
````

#### **4.3 검색 결과 개수 (top_k)**

```python
# 영향도: ⭐⭐⭐ (보통)
# 📂 src/core/rag.py
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": top_k,                    # 반환할 문서 수
        "score_threshold": 0.3,        # 최소 유사도 임계값
        "fetch_k": 50                  # 후보군 크기
    }
)

# 트레이드오프:
- top_k 증가: 더 많은 컨텍스트, 하지만 노이즈 증가
- score_threshold 상승: 높은 품질, 하지만 결과 부족 위험
```

#### 4.4 쿼리 처리 파이프라인

**영향도: ⭐⭐⭐ (보통)**

```python
# 📂 src/config/app_config.py
async def process_query_async(self, query: str, session_id: str = "default"):
    """쿼리 처리 파이프라인이 전체 답변 품질 결정"""

    try:
        # 1. 대화 히스토리 가져오기
        history = self.get_relevant_history(session_id)

        # 2. 질문 재정의
        refined_query = self.refinement_chain.run({
            "question": query,
            "context": history
        })

        # 3. 관련 문서 검색
        docs = self.qa_chain.retriever.get_relevant_documents(refined_query)

        # 4. 문서 중복 제거 및 융합
        merged_docs = self._merge_and_deduplicate_docs(docs)

        # 5. 컨텍스트 구성
        docs_text = "\n\n---\n\n".join([doc.page_content for doc in merged_docs])

        # 6. 최종 답변 생성
        result = self.qa_chain.invoke({"question": refined_query, "context": docs_text})
```

### 5. **메타데이터 품질**

#### **5.1 자동 메타데이터 추출**

```python
# 영향도: ⭐⭐⭐⭐ (높음)

def _extract_keywords(content: str) -> str:
    """키워드 품질이 검색 정확도에 직접 영향"""

    patterns = {
        '카드관련': ['신용카드', '체크카드', '발급', '해지'],
        '결제관련': ['결제', '청구', '할부', '일시불'],
        # 패턴 완성도와 정확도가 중요
    }

# 개선 필요 사항:
- 금융 전문용어 확장
- 동의어/유의어 처리
- 시간에 따른 용어 변화 대응
```

#### **5.2 카테고리 분류 정확도**

```python
# 영향도: ⭐⭐⭐ (보통)

def _extract_category(title: str, heading: str) -> str:
    """잘못된 분류는 필터링 시 문제 발생"""

    if any(keyword in combined for keyword in ['발급', '신청']):
        return "카드발급"
    elif any(keyword in combined for keyword in ['결제', '청구']):
        return "결제업무"

# 분류 오류 영향:
- 카테고리 필터 검색 시 누락
- 관련 문서 그룹핑 실패
```

### 6. **LLM 모델 성능**

#### **6.1 로컬 LLM 모델별 성능**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)

LLM_MODELS = {
    "solar_10_7b": "solar:10.7b",      # 한국어 특화, 중간 성능
    "qwen2": "qwen2:7b",               # 다국어 지원, 빠른 속도
    "llama3": "llama3:8b",             # 범용성 높음, 영어 강세
    "gemma3": "gemma3:12b"             # 큰 모델, 높은 성능
}

# 📂 src/core/models.py
def create_llm_model(model_choice: str):
    """LLM 모델 생성이 답변 품질에 결정적 영향"""

    if model_choice == "solar_10_7b":
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        model = ChatOllama(
            model=LLM_MODELS["solar_10_7b"]["model_id"],
            temperature=0,  # 일관성 있는 답변을 위해 낮은 온도
            base_url=ollama_base_url
        )

# 성능 차이 요인:
- 한국어 이해 능력
- 금융 도메인 지식
- 추론 및 요약 능력
- 표 데이터 해석 능력
```

#### **6.2 프롬프트 엔지니어링**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)

## 프롬프트
prompt_template = """
    당신은 제공된 문서만을 기반으로 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

    **[중요 규칙]**
    1. 반드시 아래 제공된 문서 내용만을 사용하여 답변해야 합니다.
    2. 문서에 없는 내용이나 사전 지식은 절대 추가하지 마세요.
    3. 질문에 관련된 문서 내용이 없으면, "죄송하지만, 제공된 문서 내에서는 질문과 관련된 정보를 찾을 수 없습니다."라고 명확하게 답변하세요.
    4. 질문이 모호하거나 여러 해석이 가능한 경우, 반드시 구체적 재질의를 요청하세요.

    **[답변 방식]**
    - 사용자의 질문 의도를 파악해, 문서에서 최대한 정확하고 친절한 답변을 제공합니다.
    - 질문이 포괄적이긴 하지만 일반 답변이 가능한 경우, 먼저 답변한 뒤 문서 기반 추가 궁금증이나 후속 질문을 2~3개 제안합니다.
    - 만약 질문이 너무 모호해 일반 답변조차 어려우면, '재질의 요청 규칙'에 따라 옵션을 제안해 질문을 구체화할 수 있도록 유도합니다.

    **[재질의 요청 규칙 및 형식]**
    - 질문이 너무 일반적이거나 여러 서비스에 걸쳐 있을 때, 혹은 핵심 키워드가 부족할 때 사용합니다.
    - 형식:
    "더 정확한 답변을 위해 다음 중 어떤 점에 대해 구체적으로 알고 싶으신지 선택해 주세요:
    1. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 1]
    2. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 2]
    3. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 3]"

    **[제공된 문서 내용]**
    {context}

    **[질문]**
    {question}

    **[답변]**
"""


## 재질의 프롬프트
prompt_for_refined_query = """
   당신은 대화 기록을 기반으로 사용자의 질문을 재정의하는 전문 AI 비서입니다.

   - **목표**: 제공된 대화 기록을 활용하여 사용자의 질문을 명확하게 다듬으세요.
   - **규칙 1**: 질문이 맥락과 관련 있다면, 이전 대화 없이도 독립적으로 이해될 수 있도록 질문을 재정의하세요.
   - **규칙 2**: 질문이 대화 맥락과 전혀 관련 없다면, 원래 질문을 그대로 반환하세요.
   - **규칙 3**: 원래 질문의 어조와 의도를 유지하세요. 평서문을 의문문으로 바꾸거나 그 반대로 바꾸지 마세요.
   - **규칙 4**: 재정의된 질문 텍스트만 출력하세요. 불필요한 단어, 문구, 대화형 표현을 일절 추가하지 마세요.
   - **규칙 5**: 원래 질문이나 대화 기록에 없는 새로운 정보나 의도를 추측하여 추가하지 마세요.
   - **규칙 6**: 답변은 반드시 한글로 합니다.

   대화 기록: {context}

   질문: {question}

   답변:
"""

## 최종 답변 프롬프트
prompt_for_query = """
   당신은 제공된 문서만을 기반으로 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

   **[중요 규칙]**
   1.  **문서 기반 답변:** 반드시 아래 제공된 문서 내용만을 사용하여 답변해야 합니다.
   2.  **외부 지식 금지:** 문서에 없는 내용이나 당신의 사전 지식은 절대 추가하지 마세요.
   3.  **정보 부재 시 답변:** 질문에 관련된 문서 내용이 없으면, "죄송하지만, 제공된 문서 내에서는 질문과 관련된 정보를 찾을 수 없습니다."라고 명확하게 답변하세요.
   4.  **모호성 처리:** 질문이 모호하거나 여러 해석이 가능하면, 아래 '재질의 요청 규칙'에 따라 반드시 구체적인 질문을 요청하세요.

   **[답변 방식]**
   1.  **답변 형식:**
      - 답변은 반드시 한국어로 작성합니다.
      - 내용을 효과적으로 전달할 수 있다면 표(테이블)를 적극적으로 활용하고, 그렇지 않다면 서술형으로 답변합니다.
      - 가독성을 높이기 위해 답변에 적절한 줄 바꿈을 사용합니다.
   2.  **포괄적 질문 대응:**
      - 사용자의 질문이 다소 포괄적이더라도 일반적인 답변이 가능한 경우, 먼저 문서에 기반해 답변합니다.
      - 그 후, 사용자가 더 궁금해할 만한 내용을 문서 기반으로 예측하여 2~3개의 구체적인 후속 질문을 제안합니다.
   3.  **모호한 질문 대응:**
      - 질문이 너무 광범위하거나 핵심 정보가 부족하여 답변이 어려울 경우, '재질의 요청 규칙'에 따라 질문을 구체화하도록 유도합니다.

   **[재질의 요청 규칙 및 형식]**
   - 질문이 너무 일반적이거나 여러 서비스에 걸쳐 있을 때 사용합니다.
   - 형식:
   "더 정확한 답변을 위해 다음 중 어떤 점에 대해 구체적으로 알고 싶으신지 선택해 주세요:
   1. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 1]
   2. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 2]
   3. [문서 내용을 바탕으로 생성한 구체적인 질문 예시 3]"

   **[제공된 문서 내용]**
   {context}

   **[질문]**
   {question}

   **[답변]**
"""

템플릿의 장점
- 명확한 역할 부여 (Persona): "정확하고 친절하게 답변하는 AI 어시스턴트"라는 역할을 명확히 하여 일관된 톤앤매너를 유지
- 강력한 제약 조건: "문서 기반", "외부 지식 금지", "정보 부재 시 답변" 규칙을 통해 환각(Hallucination)을 최소화하고 RAG의 목적에 충실
- 구체적인 출력 형식 지정: "한국어 사용", "표 활용", "줄 바꿈" 등 답변의 구체적인 형식을 지시하여 일관되고 가독성 높은 결과물을 유도
- 능동적인 사용자 경험 개선: 단순히 답변만 하는 것을 넘어, "후속 질문 제안"이나 "재질의 요청"을 통해 사용자가 원하는 정보를 더 쉽게 찾을 수 있도록 돕는 상호작용적 요소를 포함
- 체계적인 구조: 각 지시사항을 [중요 규칙], [답변 방식] 등으로 구조화하여 AI가 각 단계의 역할을 더 명확하게 이해하고 따르도록

## 요약 답변 프롬프트
prompt_for_context_summary = """
   다음 '내용'을 200자 이내로 요약하여 답변하세요.
   답변은 한국어(한글)로 작성해주세요.
   절대로 영어로 답변하지 마세요.


   내용: {context}

   답변:
"""






# 프롬프트 최적화 요소:
- 역할 정의 명확성
- 답변 규칙 구체성
- 도메인 특화 지침
- 출력 형식 가이드
```

### 7. **시스템 설정 및 파라미터**

#### **7.1 Elasticsearch 인덱스 설정**

```python
# 영향도: ⭐⭐⭐ (보통)
# 📂 src/utils/elasticsearch_index.py (추정 위치)
def create_optimized_index():
    """인덱스 설정이 검색 성능에 영향"""

    index_settings = {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": 1.2,      # 용어 빈도 가중치
                        "b": 0.75       # 문서 길이 정규화
                    }
                }
            },
            "analysis": {
                "analyzer": {
                    "korean_analyzer": {
                        "tokenizer": "nori_tokenizer",  # 한국어 형태소 분석
                        "filter": ["lowercase", "nori_part_of_speech"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": 1536,           # 임베딩 차원 (모델과 일치 필요)
                    "index": True,          # 벡터 인덱싱 활성화
                    "similarity": "cosine"  # 유사도 측정 방식
                }
            }
        }
    }
```

#### **7.2 벡터 저장소 설정**

```python
# 영향도: ⭐⭐⭐ (보통)

"vector": {
    "type": "dense_vector",
    "dims": 1536,           # 임베딩 차원 (모델과 일치 필요)
    "index": True,          # 벡터 인덱싱 활성화
    "similarity": "cosine"  # 유사도 측정 방식
}
```

### 8. **데이터 품질 요소**

#### **8.1 원본 문서 품질**

```python
# 영향도: ⭐⭐⭐⭐⭐ (매우 높음)

# PDF 품질 요소:
- 스캔 vs 네이티브 PDF
- 표 구조의 복잡성
- 폰트 및 레이아웃
- 이미지 포함 여부

# JSON 구조화 정도:
- 메타데이터 완성도 (title, heading, section)
- 내용 분할의 적절성
- hasTable 정보 정확성
```

#### **8.2 도메인 특화 정보**

```python
# 영향도: ⭐⭐⭐⭐ (높음)

# 금융 용어 처리:
- 전문 용어 사전 구축
- 약어 및 동의어 매핑
- 법적 용어 정확성
- 수치 정보 처리

# 시간 민감 정보:
- 금리, 수수료 변경
- 정책 업데이트
- 계절별 이벤트
```

## 🎯 정확도 개선 우선순위

### **1 우선순위 (⭐⭐⭐⭐⭐)**

1. **표 데이터 처리 개선**: `_is_table_content()` 함수 정확도 향상
2. **프롬프트 최적화**: 도메인 특화 프롬프트 개발
3. **LLM 모델 선택**: 한국어 금융 도메인 최적화
4. **원본 데이터 품질**: PDF 전처리, JSON 구조화 개선

**📂 elasticsearch_index.py**

- `_is_table_content()`: 표 감지 정확도 향상
- `_extract_keywords()`: 키워드 패턴 확장
- `_enhanced_pdf_loader()`: PDF 처리 품질 개선

**📂 rag.py**

- `custom_prompt`: 프롬프트 최적화
- RRF 파라미터 조정

**📂 models.py**

- `create_llm_model()`: 한국어 특화 모델 선택

### **2 우선순위 (⭐⭐⭐⭐)**

1. **하이브리드 검색 가중치 조정**: 필드별 boost 값 최적화
2. **청킹 전략 개선**: chunk_size, overlap 최적화
3. **메타데이터 품질**: 키워드 추출, 카테고리 분류 정확도
4. **임베딩 모델 선택**: 한국어 성능 vs 비용 고려

**📂 elasticsearch.py**

- `keyword_search()`: 가중치 최적화
- `_combine_search_results()`: RRF 알고리즘 개선

**📂 app_config.py**

- `create_refinement_chain()`: 질문 재정의 로직
- `_merge_and_deduplicate_docs()`: 문서 융합 품질

**📂 elasticsearch_index.py**

- `get_optimized_text_splitter()`: 청킹 파라미터 조정

### **3 우선순위 (⭐⭐⭐)**

1. **검색 파라미터 튜닝**: top_k, score_threshold 조정
2. **인덱스 설정 최적화**: BM25 파라미터, 분석기 설정
3. **답변 품질 모니터링**: 자동화된 품질 측정 시스템
