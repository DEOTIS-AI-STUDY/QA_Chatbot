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
from core.chat_history import ChatHistoryManager, CLIChatHistoryInterface
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
        
        # 공통 대화 기록 관리자 사용
        self.chat_manager = ChatHistoryManager(max_history=10)
        self.chat_interface = CLIChatHistoryInterface(self.chat_manager)
        
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
            is_connected, connection_msg = es_manager.check_connection()
            if not is_connected:
                issues.append(f"❌ Elasticsearch 서버 연결 실패: {connection_msg}")
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
    
    def _auto_index_pdfs(self) -> bool:
        """PDF 파일 자동 인덱싱"""
        from core.config import PDF_DIR, INDEX_NAME
        from elasticsearch import Elasticsearch
        
        # PDF 디렉토리 확인
        if not os.path.exists(PDF_DIR):
            print(f"📁 PDF 디렉토리({PDF_DIR})가 없습니다. 생성합니다...")
            os.makedirs(PDF_DIR, exist_ok=True)
            print("📄 PDF 파일이 없어서 인덱싱을 건너뜁니다.")
            return True
        
        # PDF 파일 목록 확인
        pdf_files = self.es_manager.list_pdfs(PDF_DIR)
        if not pdf_files:
            print("📄 PDF 파일이 없어서 인덱싱을 건너뜁니다.")
            return True
        
        # 인덱스 존재 확인
        es = Elasticsearch("http://localhost:9200")
        index_exists = es.indices.exists(index=INDEX_NAME)
        
        if index_exists:
            # 문서 수 확인
            try:
                doc_count = es.count(index=INDEX_NAME).get("count", 0)
                if doc_count > 0:
                    print(f"📚 기존 인덱스에 {doc_count}개 문서가 있습니다. 인덱싱을 건너뜁니다.")
                    return True
            except:
                pass
        
        # PDF 파일 인덱싱 실행
        print(f"📄 {len(pdf_files)}개 PDF 파일을 자동 인덱싱합니다...")
        for pdf_file in pdf_files:
            print(f"  - {os.path.basename(pdf_file)}")
        
        try:
            # 간단한 트래커 생성 (CLI용)
            class SimpleTracker:
                def track_preprocessing_stage(self, stage): pass
                def end_preprocessing_stage(self, stage): pass
            
            tracker = SimpleTracker()
            success, message = self.es_manager.index_pdfs(pdf_files, self.embedding_model, tracker)
            
            if success:
                print(f"✅ PDF 자동 인덱싱 완료: {message}")
                return True
            else:
                print(f"❌ PDF 자동 인덱싱 실패: {message}")
                return False
                
        except Exception as e:
            print(f"❌ PDF 자동 인덱싱 오류: {str(e)}")
            return False
    
    def initialize_rag_system(self, init_index: bool = False) -> bool:
        """RAG 시스템 초기화"""
        print("\n🚀 RAG 시스템 초기화 중...")
        
        # 1. Elasticsearch 연결
        try:
            self.es_manager = ElasticsearchManager()
            is_connected, connection_msg = self.es_manager.check_connection()
            if not is_connected:
                print(f"❌ Elasticsearch 연결 실패: {connection_msg}")
                return False
            print(f"✅ Elasticsearch 연결 성공: {connection_msg}")
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
        
        # 2.5. PDF 자동 인덱싱 (--init-index 옵션이 있을 때만)
        if init_index:
            try:
                success = self._auto_index_pdfs()
                if not success:
                    print("⚠️ PDF 자동 인덱싱 실패, 계속 진행합니다...")
            except Exception as e:
                print(f"⚠️ PDF 자동 인덱싱 오류: {str(e)}, 계속 진행합니다...")
        else:
            print("📄 PDF 자동 인덱싱 건너뜀 (--init-index 옵션 사용 시 실행)")
        
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
        """질의 처리 - 공통 대화 기록 관리자 사용"""
        if not self.rag_chain:
            print("❌ RAG 시스템이 초기화되지 않았습니다.")
            return None
        
        try:
            print(f"\n🔍 질의 처리 중: {query}")
            
            # 대화 기록을 포함한 컨텍스트 구성
            context_query = self.chat_manager.build_context_query(query)
            
            # RAG 체인 실행
            response = self.rag_chain({"query": context_query})
            
            # 응답 처리
            answer = self._extract_answer(response)
            
            if answer:
                # 공통 대화 기록 관리자에 추가 (원본 질문과 답변만)
                self.chat_manager.add_chat(query, answer)
                return answer
            
            return None
                
        except Exception as e:
            print(f"❌ 질의 처리 오류: {str(e)}")
            return None
    
    def _extract_answer(self, response) -> Optional[str]:
        """응답에서 답변 추출"""
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
    
    def interactive_chat(self):
        """대화형 채팅 모드 - 대화 기록 포함"""
        self.show_system_info()
        
        print("\n💬 대화형 채팅 모드 시작")
        print("(종료: 'exit', 'quit', 또는 Ctrl+C)")
        print("(대화기록 보기: 'history', 기록 삭제: 'clear')")
        print("-" * 60)
        
        chat_count = 0
        
        while True:
            try:
                # 사용자 입력
                query = input(f"\n[{chat_count + 1}] 질문을 입력하세요: ").strip()
                
                if not query:
                    continue
                
                # 특수 명령어 처리
                if query.lower() in ['exit', 'quit', '종료']:
                    print("\n👋 채팅을 종료합니다.")
                    break
                elif query.lower() in ['history', '기록', '히스토리']:
                    self.chat_interface.show_history()
                    continue
                elif query.lower() in ['clear', '삭제', '클리어']:
                    self.chat_interface.clear_history_with_confirmation()
                    continue
                
                # 질의 처리
                response = self.process_query(query)
                
                if response:
                    chat_count += 1
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n🤖 답변 [{timestamp}]:")
                    print("-" * 40)
                    
                    # 텍스트 래핑으로 가독성 향상
                    wrapped_response = textwrap.fill(response, width=80)
                    print(wrapped_response)
                    print("-" * 40)
                    
                    # 대화 기록 요약 표시
                    if self.chat_manager.has_history():
                        self.chat_interface.show_status_info()
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
          python unified_rag_cli.py --init-index              # PDF 자동 인덱싱 후 대화형 모드
          python unified_rag_cli.py --model upstage          # 모델 지정 후 대화형 모드  
          python unified_rag_cli.py --query "질문 내용"       # 단일 질의 모드
          python unified_rag_cli.py --model solar_10_7b --init-index --query "질문"
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
        "--init-index",
        action="store_true",
        help="PDF 파일 자동 인덱싱 실행 (최초 실행 시에만 사용)"
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
    if not rag_system.initialize_rag_system(args.init_index):
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
