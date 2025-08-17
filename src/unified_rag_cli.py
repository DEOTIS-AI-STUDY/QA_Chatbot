#!/usr/bin/env python3
"""
CLI 기반 통합 RAG 시스템
- 기존 unified_rag_refactored.py의 core 로직 재사용
- 대화형 CLI 인터페이스 제공
- 모델 선택 및 RAG 질의 응답
"""

import os
import sys
import argparse
import textwrap
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 프로젝트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 기존 모듈 임포트
from core.config import LLM_MODELS, HUGGINGFACE_EMBEDDINGS_AVAILABLE, UPSTAGE_AVAILABLE, OLLAMA_AVAILABLE
from core.models import ModelFactory
from core.rag import create_rag_chain
from utils.elasticsearch import ElasticsearchManager

# Elasticsearch 가용성 확인
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# 환경 변수 로드
load_dotenv()

class CLIRAGSystem:
    """CLI 기반 RAG 시스템"""
    
    def __init__(self):
        self.es_manager = None
        self.model_factory = ModelFactory()
        self.rag_chain = None
        self.embedding_model = None
        self.llm_model = None
        self.model_choice = None
        self.top_k = 5
        
    def check_dependencies(self) -> bool:
        """의존성 확인"""
        print("🔍 시스템 의존성 확인 중...")
        
        issues = []
        
        if not ELASTICSEARCH_AVAILABLE:
            issues.append("❌ Elasticsearch 라이브러리가 설치되지 않았습니다.")
        
        if not HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            issues.append("❌ HuggingFace 임베딩 라이브러리가 설치되지 않았습니다.")
        
        # Elasticsearch 서버 연결 확인
        try:
            es_manager = ElasticsearchManager()
            if not es_manager.check_connection():
                issues.append("❌ Elasticsearch 서버에 연결할 수 없습니다. (http://localhost:9200)")
        except Exception as e:
            issues.append(f"❌ Elasticsearch 연결 오류: {str(e)}")
        
        if issues:
            print("\n".join(issues))
            print("\n해결 방법:")
            print("1. Python 패키지 설치:")
            print("   pip install -r requirements.txt")
            print("   또는: pip install -r requirements_unified.txt")
            print("2. Elasticsearch 서버 시작:")
            print("   ./setup.sh               # Docker + Elasticsearch 자동 시작")
            print("   또는: ./start.sh         # 전체 시스템 자동 시작")
            print("   또는: docker-compose up -d elasticsearch")
            print("3. Ollama 모델 확인:")
            print("   ollama list              # 설치된 모델 확인")
            print("   ollama pull solar:10.7b  # SOLAR 모델 설치")
            return False
        
        print("✅ 모든 의존성이 정상입니다.")
        return True
    
    def show_available_models(self) -> Dict[str, Any]:
        """사용 가능한 모델 목록 표시"""
        print("\n📋 사용 가능한 LLM 모델:")
        
        available_models = self.model_factory.get_available_models()
        
        if not available_models:
            print("❌ 사용 가능한 모델이 없습니다.")
            return {}
        
        model_list = []
        for i, (key, config) in enumerate(available_models.items(), 1):
            print(f"{i}. {config['name']} (키: {key})")
            model_list.append((key, config))
        
        return dict(model_list)
    
    def select_model(self, model_key: Optional[str] = None) -> bool:
        """모델 선택"""
        available_models = self.show_available_models()
        
        if not available_models:
            return False
        
        if model_key:
            # CLI 인자로 모델이 지정된 경우
            if model_key in available_models:
                self.model_choice = model_key
                print(f"✅ 선택된 모델: {available_models[model_key]['name']}")
                return True
            else:
                print(f"❌ 모델 '{model_key}'를 찾을 수 없습니다.")
                return False
        
        # 대화형 모델 선택
        while True:
            try:
                choice = input("\n모델 번호를 선택하세요 (1-{}): ".format(len(available_models)))
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_models):
                    model_keys = list(available_models.keys())
                    self.model_choice = model_keys[choice_idx]
                    selected_model = available_models[self.model_choice]
                    print(f"✅ 선택된 모델: {selected_model['name']}")
                    return True
                else:
                    print("❌ 잘못된 번호입니다.")
            except (ValueError, KeyboardInterrupt):
                print("\n프로그램을 종료합니다.")
                return False
    
    def set_search_parameters(self, top_k: Optional[int] = None):
        """검색 매개변수 설정"""
        if top_k:
            self.top_k = top_k
            print(f"✅ Top-K 설정: {self.top_k}")
            return
        
        print(f"\n현재 Top-K 설정: {self.top_k}")
        try:
            new_k = input("새로운 Top-K 값을 입력하세요 (엔터: 기본값 유지): ").strip()
            if new_k:
                self.top_k = int(new_k)
                print(f"✅ Top-K가 {self.top_k}로 설정되었습니다.")
        except ValueError:
            print("❌ 잘못된 값입니다. 기본값을 유지합니다.")
    
    def initialize_rag_system(self) -> bool:
        """RAG 시스템 초기화"""
        print("\n🚀 RAG 시스템 초기화 중...")
        
        # 1. Elasticsearch 연결
        try:
            self.es_manager = ElasticsearchManager()
            if not self.es_manager.check_connection():
                print("❌ Elasticsearch 연결 실패")
                return False
            print("✅ Elasticsearch 연결 성공")
        except Exception as e:
            print(f"❌ Elasticsearch 초기화 오류: {str(e)}")
            return False
        
        # 2. 임베딩 모델 로드
        try:
            self.embedding_model = self.model_factory.create_embedding_model()
            if not self.embedding_model:
                print("❌ 임베딩 모델 로드 실패")
                return False
            print("✅ BGE-M3 임베딩 모델 로드 성공")
        except Exception as e:
            print(f"❌ 임베딩 모델 오류: {str(e)}")
            return False
        
        # 3. LLM 모델 로드
        try:
            self.llm_model, status = self.model_factory.create_llm_model(self.model_choice)
            if not self.llm_model:
                print(f"❌ LLM 모델 로드 실패: {status}")
                return False
            print(f"✅ {status}")
        except Exception as e:
            print(f"❌ LLM 모델 오류: {str(e)}")
            return False
        
        # 4. RAG 체인 생성
        try:
            self.rag_chain, success = create_rag_chain(
                embeddings=self.embedding_model,
                llm_model=self.llm_model,
                top_k=self.top_k
            )
            if not self.rag_chain:
                print(f"❌ RAG 체인 생성 실패: {success}")
                return False
            print("✅ RAG 체인 생성 성공")
        except Exception as e:
            print(f"❌ RAG 체인 오류: {str(e)}")
            return False
        
        print("\n🎉 RAG 시스템 초기화 완료!")
        return True
    
    def show_system_info(self):
        """시스템 정보 표시"""
        print("\n" + "="*60)
        print("🤖 통합 RAG 시스템 - CLI 모드")
        print("="*60)
        print(f"📋 선택된 모델: {LLM_MODELS[self.model_choice]['name']}")
        print(f"🔍 검색 결과 수: Top-{self.top_k}")
        print(f"🗄️ 임베딩: BGE-M3")
        print(f"🔗 벡터 DB: Elasticsearch (localhost:9200)")
        print("="*60)
    
    def process_query(self, query: str) -> Optional[str]:
        """질의 처리"""
        if not self.rag_chain:
            print("❌ RAG 시스템이 초기화되지 않았습니다.")
            return None
        
        try:
            print(f"\n🔍 질의 처리 중: {query}")
            
            # RAG 체인 실행 - core/rag.py에서 사용하는 형식에 맞춤
            response = self.rag_chain({"query": query})
            
            # response는 딕셔너리 형태일 것으로 예상됨
            if isinstance(response, dict):
                if 'result' in response:
                    return response['result']
                elif 'answer' in response:
                    return response['answer']
                else:
                    return str(response)
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            print(f"❌ 질의 처리 오류: {str(e)}")
            return None
    
    def interactive_chat(self):
        """대화형 채팅 모드"""
        self.show_system_info()
        
        print("\n💬 대화형 채팅 모드 시작")
        print("(종료: 'exit', 'quit', 또는 Ctrl+C)")
        print("-" * 60)
        
        chat_count = 0
        
        while True:
            try:
                # 사용자 입력
                query = input(f"\n[{chat_count + 1}] 질문을 입력하세요: ").strip()
                
                if not query:
                    continue
                
                # 종료 명령어 확인
                if query.lower() in ['exit', 'quit', '종료']:
                    print("\n👋 채팅을 종료합니다.")
                    break
                
                # 질의 처리
                response = self.process_query(query)
                
                if response:
                    chat_count += 1
                    print(f"\n🤖 답변:")
                    print("-" * 40)
                    
                    # 텍스트 래핑으로 가독성 향상
                    wrapped_response = textwrap.fill(response, width=80)
                    print(wrapped_response)
                    print("-" * 40)
                else:
                    print("❌ 답변을 생성할 수 없습니다.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
    
    def single_query_mode(self, query: str):
        """단일 질의 모드"""
        self.show_system_info()
        
        print(f"\n🔍 질의: {query}")
        response = self.process_query(query)
        
        if response:
            print(f"\n🤖 답변:")
            print("="*60)
            wrapped_response = textwrap.fill(response, width=80)
            print(wrapped_response)
            print("="*60)
        else:
            print("❌ 답변을 생성할 수 없습니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="CLI 기반 통합 RAG 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        사용 예시:
          python unified_rag_cli.py                           # 대화형 모드
          python unified_rag_cli.py --model upstage          # 모델 지정 후 대화형 모드  
          python unified_rag_cli.py --query "질문 내용"       # 단일 질의 모드
          python unified_rag_cli.py --model solar_10_7b --query "질문" --top-k 10
        ''')
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="사용할 LLM 모델 (upstage, solar_10_7b, qwen2, llama3)"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="질의 텍스트 (단일 질의 모드)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="검색 결과 상위 K개 (기본값: 5)"
    )
    
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="의존성 검사 건너뛰기"
    )
    
    args = parser.parse_args()
    
    # CLI RAG 시스템 초기화
    rag_system = CLIRAGSystem()
    
    # 의존성 확인
    if not args.no_check:
        if not rag_system.check_dependencies():
            print("\n의존성 문제로 프로그램을 종료합니다.")
            sys.exit(1)
    
    # 모델 선택
    if not rag_system.select_model(args.model):
        print("\n모델 선택에 실패했습니다.")
        sys.exit(1)
    
    # 검색 매개변수 설정
    rag_system.set_search_parameters(args.top_k)
    
    # RAG 시스템 초기화
    if not rag_system.initialize_rag_system():
        print("\nRAG 시스템 초기화에 실패했습니다.")
        sys.exit(1)
    
    # 실행 모드 결정
    if args.query:
        # 단일 질의 모드
        rag_system.single_query_mode(args.query)
    else:
        # 대화형 모드
        rag_system.interactive_chat()


if __name__ == "__main__":
    main()
