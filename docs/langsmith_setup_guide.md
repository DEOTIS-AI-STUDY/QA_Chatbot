# 🔑 Langsmith API 키 설정 완전 가이드

## 📋 **단계별 설정 방법**

### **Step 1: Langsmith 계정 및 API 키 발급**

1. **웹사이트 접속**: [https://smith.langchain.com/](https://smith.langchain.com/)
2. **계정 생성**: Sign Up (GitHub/Google 연동 가능)
3. **API 키 발급**:
   - 로그인 후 우측 상단 프로필 → Settings
   - 좌측 메뉴 "API Keys" → "Create API Key"
   - 키 이름 입력 (예: "PDF-QA-Chatbot")
   - **API 키 복사** (중요: 한 번만 표시됨!)

### **Step 2: API 키 설정 (3가지 방법 중 선택)**

#### **🌟 방법 1: 환경변수 설정 (가장 추천)**

**임시 설정 (현재 터미널 세션만)**:

```bash
export LANGSMITH_API_KEY="lsv2_pt_여기에_실제_API키_입력"
export LANGSMITH_PROJECT="pdf-qa-hybrid-monitoring"
```

**영구 설정 (재부팅 후에도 유지)**:

```bash
# zsh 사용자 (macOS 기본)
echo 'export LANGSMITH_API_KEY="lsv2_pt_여기에_실제_API키_입력"' >> ~/.zshrc
echo 'export LANGSMITH_PROJECT="pdf-qa-hybrid-monitoring"' >> ~/.zshrc
source ~/.zshrc

# bash 사용자
echo 'export LANGSMITH_API_KEY="lsv2_pt_여기에_실제_API키_입력"' >> ~/.bashrc
echo 'export LANGSMITH_PROJECT="pdf-qa-hybrid-monitoring"' >> ~/.bashrc
source ~/.bashrc
```

#### **📁 방법 2: .env 파일 사용**

```bash
# 1. 템플릿 복사
cp .env.example .env

# 2. .env 파일 편집
nano .env
# 또는
code .env
```

**.env 파일 내용**:

```bash
LANGSMITH_API_KEY=lsv2_pt_여기에_실제_API키_입력
LANGSMITH_PROJECT=pdf-qa-hybrid-monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

#### **🔒 방법 3: Streamlit Secrets**

```bash
# 1. 디렉토리 생성
mkdir -p .streamlit

# 2. secrets 파일 생성
nano .streamlit/secrets.toml
```

**.streamlit/secrets.toml 내용**:

```toml
LANGSMITH_API_KEY = "lsv2_pt_여기에_실제_API키_입력"
LANGSMITH_PROJECT = "pdf-qa-hybrid-monitoring"
```

### **Step 3: 설정 확인**

```bash
# 설정 확인 도구 실행
./check_langsmith.sh
```

**출력 예시**:

```
🔑 Langsmith API 키 설정 확인 도구
==================================
1. 환경변수 확인:
   ✅ LANGSMITH_API_KEY: 설정됨 (lsv2_pt_ab...)
   ✅ LANGSMITH_PROJECT: pdf-qa-hybrid-monitoring

2. .env 파일 확인:
   ✅ .env 파일 존재
   ✅ .env에 LANGSMITH_API_KEY 설정됨
```

### **Step 4: 애플리케이션 실행**

```bash
./run.sh
```

### **🎯 설정 성공 확인 방법**

애플리케이션 실행 후 다음을 확인하세요:

1. **메인 화면**: "✅ Langsmith 추적 활성화됨" 메시지
2. **사이드바**: "📊 Langsmith: ✅" 상태 표시
3. **질문 응답**: "📊 Langsmith LLM 추적: ✅" 표시
4. **웹 대시보드**: [Langsmith 대시보드](https://smith.langchain.com/)에서 추적 로그 확인

### **❗ 문제 해결**

#### **API 키가 인식되지 않는 경우**:

1. API 키 형식 확인 (`lsv2_pt_`로 시작해야 함)
2. 따옴표 포함 여부 확인
3. 터미널 재시작 후 다시 시도

#### **Langsmith 라이브러리 오류**:

```bash
# 라이브러리 재설치
pip install langsmith>=0.1.0
```

#### **권한 오류**:

```bash
# 스크립트 실행 권한 부여
chmod +x check_langsmith.sh
chmod +x run.sh
```

### **🔐 보안 주의사항**

- ✅ API 키를 Git에 커밋하지 마세요
- ✅ `.env` 파일을 `.gitignore`에 추가하세요
- ✅ API 키는 안전한 곳에 백업하세요
- ✅ 불필요한 권한은 부여하지 마세요

---

## 🚀 **완료 후 혜택**

✅ **LLM 호출 추적**: 모든 질문과 응답이 자동 로깅
✅ **토큰 사용량 분석**: 비용 최적화 가능
✅ **응답 품질 평가**: 답변 개선점 파악
✅ **팀 협업**: 웹 대시보드로 성능 공유
✅ **장기 분석**: 사용 패턴과 트렌드 분석

🌐 Langsmith Endpoint 옵션들

1. 🏢 Enterprise 옵션들
   A. LangSmith Cloud (공식 호스팅)
   공식 LangChain 팀에서 운영
   가장 안정적이고 최신 기능 제공
   B. LangSmith Enterprise Cloud
   기업 전용 클라우드 인스턴스
   더 높은 보안 및 격리
   C. Self-Hosted Enterprise
   자체 인프라에서 호스팅
   완전한 데이터 제어
2. 🔧 개발/테스트 환경
   A. 로컬 개발 서버
   B. 내부 테스트 서버
   C. 스테이징 환경
3. 🎯 실제 사용 사례들
   Read .env.example

4. 🔐 보안 및 규정 준수
   기업용 요구사항에 따른 선택
5. 🌍 지역별 서버
   💡 현재 설정 확인
   🎯 정리
   현재는 공식 서버(api.smith.langchain.com)를 사용하고 있지만, 필요에 따라 변경 가능합니다:

✅ 개인/소규모: 공식 서버 (현재 설정)
🏢 기업용: Enterprise Cloud 또는 Self-hosted
🔧 개발용: 로컬 또는 내부 서버
🌍 글로벌: 지역별 서버
변경하려면 .env 파일의 LANGCHAIN_ENDPOINT만 수정하면 됩니다! 🚀
