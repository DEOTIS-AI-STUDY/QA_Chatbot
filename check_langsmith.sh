#!/bin/bash

# Langsmith API 키 설정 확인 도구

echo "🔑 Langsmith API 키 설정 확인 도구"
echo "=================================="

# 환경변수 확인
echo "1. 환경변수 확인:"
if [ -n "$LANGSMITH_API_KEY" ]; then
    echo "   ✅ LANGSMITH_API_KEY: 설정됨 (${LANGSMITH_API_KEY:0:10}...)"
else
    echo "   ❌ LANGSMITH_API_KEY: 설정되지 않음"
fi

if [ -n "$LANGSMITH_PROJECT" ]; then
    echo "   ✅ LANGSMITH_PROJECT: $LANGSMITH_PROJECT"
else
    echo "   ❌ LANGSMITH_PROJECT: 설정되지 않음"
fi

echo ""

# .env 파일 확인
echo "2. .env 파일 확인:"
if [ -f ".env" ]; then
    echo "   ✅ .env 파일 존재"
    if grep -q "LANGSMITH_API_KEY" .env; then
        echo "   ✅ .env에 LANGSMITH_API_KEY 설정됨"
    else
        echo "   ❌ .env에 LANGSMITH_API_KEY 없음"
    fi
else
    echo "   ❌ .env 파일 없음"
fi

echo ""

# Streamlit secrets 확인
echo "3. Streamlit secrets 확인:"
if [ -f ".streamlit/secrets.toml" ]; then
    echo "   ✅ secrets.toml 파일 존재"
    if grep -q "LANGSMITH_API_KEY" .streamlit/secrets.toml; then
        echo "   ✅ secrets.toml에 LANGSMITH_API_KEY 설정됨"
    else
        echo "   ❌ secrets.toml에 LANGSMITH_API_KEY 없음"
    fi
else
    echo "   ❌ secrets.toml 파일 없음"
fi

echo ""

# 설정 방법 안내
echo "4. 설정 방법 (선택하세요):"
echo ""
echo "방법 1 - 환경변수 설정 (추천):"
echo "export LANGSMITH_API_KEY='your_api_key_here'"
echo "export LANGSMITH_PROJECT='pdf-qa-hybrid-monitoring'"
echo ""
echo "방법 2 - .env 파일 설정:"
echo "cp .env.example .env"
echo "# .env 파일을 편집하여 API 키 입력"
echo ""
echo "방법 3 - Streamlit secrets 설정:"
echo "mkdir -p .streamlit"
echo "# .streamlit/secrets.toml 파일을 생성하여 API 키 입력"
echo ""

# API 키 유효성 간단 체크
if [ -n "$LANGSMITH_API_KEY" ]; then
    echo "5. API 키 형식 확인:"
    if [[ $LANGSMITH_API_KEY == lsv2_pt_* ]]; then
        echo "   ✅ API 키 형식이 올바름 (lsv2_pt_로 시작)"
    else
        echo "   ⚠️  API 키 형식 확인 필요 (보통 lsv2_pt_로 시작)"
    fi
fi

echo ""
echo "📋 완료! ./run.sh를 실행하여 하이브리드 추적을 확인하세요."
