# FastAPI RAG 시스템

CLI와 Streamlit 기반의 RAG 시스템을 FastAPI로 변환한 RESTful API 서버입니다.

## 📋 주요 특징

- **RESTful API**: 표준 HTTP 메서드를 사용한 API 제공
- **비동기 처리**: FastAPI의 비동기 기능 활용
- **세션 관리**: 다중 사용자 세션별 대화 기록 관리
- **웹 인터페이스**: 간단한 HTML/JS 기반 웹 UI 제공
- **Docker 지원**: 컨테이너화된 배포 환경
- **자동 문서화**: OpenAPI/Swagger 자동 생성
- **CORS 지원**: 크로스 오리진 요청 허용

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# FastAPI 관련 패키지 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 개발 모드 (자동 리로드)
./run.sh
export RELOAD=true && ./run.sh

# 프로덕션 모드
python main.py
```

### 3. API 문서 확인

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json

### 4. 웹 인터페이스 사용

`web.html` 파일을 브라우저에서 열면 간단한 웹 인터페이스를 사용할 수 있습니다.

## 📚 API 엔드포인트

### 시스템 관리

| 메서드 | 엔드포인트      | 설명                  |
| ------ | --------------- | --------------------- |
| GET    | `/`             | 루트 정보             |
| GET    | `/health`       | 헬스체크              |
| GET    | `/dependencies` | 의존성 확인           |
| GET    | `/models`       | 사용 가능한 모델 목록 |
| POST   | `/initialize`   | RAG 시스템 초기화     |
| GET    | `/status`       | 시스템 상태 확인      |

### 질의 응답

| 메서드 | 엔드포인트 | 설명      |
| ------ | ---------- | --------- |
| POST   | `/query`   | 질의 처리 |

### 대화 기록 관리

| 메서드 | 엔드포인트                   | 설명           |
| ------ | ---------------------------- | -------------- |
| GET    | `/chat/history/{session_id}` | 대화 기록 조회 |
| DELETE | `/chat/history/{session_id}` | 대화 기록 삭제 |

### 세션 관리

| 메서드 | 엔드포인트               | 설명           |
| ------ | ------------------------ | -------------- |
| GET    | `/sessions`              | 활성 세션 목록 |
| DELETE | `/sessions/{session_id}` | 세션 삭제      |

## 🔧 사용 예제

### Python 클라이언트

```python
import asyncio
from client import RAGAPIClient

async def main():
    async with RAGAPIClient() as client:
        # 모델 목록 확인
        models = await client.get_available_models()
        print(f"사용 가능한 모델: {list(models['models'].keys())}")

        # 시스템 초기화
        model_name = list(models['models'].keys())[0]
        result = await client.initialize_system(model_name, top_k=5)
        print(f"초기화: {result}")

        # 질의 처리
        response = await client.query("BC카드의 주요 서비스는 무엇인가요?")
        print(f"답변: {response['answer']}")

asyncio.run(main())
```

### cURL 예제

```bash
# 헬스체크
curl http://127.0.0.1:8000/health

# 모델 목록
curl http://127.0.0.1:8000/models

# 시스템 초기화
curl -X POST http://127.0.0.1:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"model": "upstage", "top_k": 5}'

# 질의 처리
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "BC카드의 주요 서비스는?", "session_id": "test"}'
```

### JavaScript/Fetch 예제

```javascript
// 시스템 초기화
const initResponse = await fetch("http://127.0.0.1:8000/initialize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model: "upstage", top_k: 5 }),
});

// 질의 처리
const queryResponse = await fetch("http://127.0.0.1:8000/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "BC카드 서비스 문의",
    session_id: "web_session",
  }),
});

const result = await queryResponse.json();
console.log(result.answer);
```

## 🐳 Docker 배포

### 단독 실행

```bash
# 이미지 빌드
docker build -t rag-api .

# 컨테이너 실행
docker run -p 8000:8000 rag-api
```

### Docker Compose

```bash
# 전체 스택 실행 (Elasticsearch + RAG API + Nginx)
docker-compose up -d

# 로그 확인
docker-compose logs -f rag-api

# 중지
docker-compose down
```

## ⚙️ 환경 설정

### 환경 변수

| 변수                | 기본값                  | 설명                  |
| ------------------- | ----------------------- | --------------------- |
| `HOST`              | `127.0.0.1`             | 서버 바인딩 주소      |
| `PORT`              | `8000`                  | 서버 포트             |
| `RELOAD`            | `false`                 | 개발 모드 자동 리로드 |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch URL     |

### 설정 파일

기존 `core/config.py`의 설정을 그대로 사용합니다.

## 🔒 보안 고려사항

### 프로덕션 배포 시 주의사항

1. **CORS 설정**: `allow_origins`을 특정 도메인으로 제한
2. **API 키 인증**: 필요시 인증 미들웨어 추가
3. **HTTPS**: 리버스 프록시를 통한 SSL 종료
4. **Rate Limiting**: 요청 제한 미들웨어 적용

```python
# 예: CORS 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 성능 최적화

### 메모리 사용량

- 세션별 대화 기록은 메모리에 저장됩니다
- 장기간 실행 시 메모리 사용량 모니터링 필요
- 필요시 Redis 등 외부 스토리지 사용 고려

### 동시 요청 처리

- FastAPI의 비동기 기능으로 다중 요청 처리
- CPU 집약적 작업은 백그라운드 태스크로 처리
- 필요시 워커 프로세스 수 조정

## 🐛 문제 해결

### 일반적인 문제들

1. **Elasticsearch 연결 실패**

   ```bash
   # Elasticsearch 서버 시작
   docker-compose up -d elasticsearch
   ```

2. **Ollama 연결 실패**

   ```bash
   # Ollama 서버 시작
   ollama serve
   ```

3. **모델 로드 실패**
   - 충분한 메모리 확보
   - 모델 파일 다운로드 확인

### 로그 확인

```bash
# 개발 모드에서 상세 로그 확인
uvicorn main:app --host 127.0.0.1 --port 8000 --log-level debug
```

## 🔄 CLI/Streamlit에서 FastAPI로 변경사항

### 주요 변화

1. **인터페이스**: CLI/UI → RESTful API
2. **세션 관리**: 단일 세션 → 다중 세션 지원
3. **상태 관리**: 전역 상태 → 요청별 상태
4. **에러 처리**: 예외 출력 → HTTP 상태 코드
5. **배포**: 로컬 실행 → 서버 배포

### 유지된 기능

- Core RAG 로직 (models, rag, chat_history)
- Elasticsearch 연동
- 다중 LLM 모델 지원
- 임베딩 및 검색 기능
- 성능 측정

## 🤝 기여

버그 리포트나 기능 요청은 이슈로 등록해주세요.

## 📝 라이선스

기존 프로젝트와 동일한 라이선스를 따릅니다.
