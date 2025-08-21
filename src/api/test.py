#!/usr/bin/env python3
"""
FastAPI RAG 시스템 테스트 스크립트
"""

import asyncio
import sys
import time
from typing import Dict, Any

# 클라이언트 모듈 임포트
try:
    from client import RAGAPIClient
except ImportError:
    print("❌ client.py 모듈을 찾을 수 없습니다.")
    sys.exit(1)


class RAGAPITester:
    """RAG API 테스터"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8110"):
        self.base_url = base_url
        self.test_session_id = f"test_session_{int(time.time())}"
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_result(self, test_name: str, passed: bool, message: str = ""):
        """테스트 결과 출력"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASS {message}")
        else:
            print(f"❌ {test_name}: FAIL {message}")
    
    async def test_basic_endpoints(self, client: RAGAPIClient):
        """기본 엔드포인트 테스트"""
        print("\n🔍 기본 엔드포인트 테스트")
        print("-" * 40)
        
        # 1. 헬스체크
        try:
            health = await client.health_check()
            self.test_result(
                "헬스체크", 
                health.get('status') == 'healthy',
                f"응답: {health}"
            )
        except Exception as e:
            self.test_result("헬스체크", False, f"예외: {e}")
        
        # 2. 모델 목록
        try:
            models = await client.get_available_models()
            has_models = 'models' in models and len(models['models']) > 0
            self.test_result(
                "모델 목록", 
                has_models,
                f"모델 수: {len(models.get('models', {}))}"
            )
            return models.get('models', {})
        except Exception as e:
            self.test_result("모델 목록", False, f"예외: {e}")
            return {}
        
    async def test_dependency_check(self, client: RAGAPIClient):
        """의존성 확인 테스트"""
        print("\n🔧 의존성 확인 테스트")
        print("-" * 40)
        
        try:
            deps = await client.check_dependencies()
            status_ok = deps.get('status') == 'ok'
            self.test_result(
                "의존성 확인",
                status_ok,
                f"상태: {deps.get('status')}, 문제: {len(deps.get('issues', []))}"
            )
            
            if not status_ok:
                print("   의존성 문제:")
                for issue in deps.get('issues', []):
                    print(f"   - {issue}")
                    
        except Exception as e:
            self.test_result("의존성 확인", False, f"예외: {e}")
    
    async def test_system_initialization(self, client: RAGAPIClient, models: Dict[str, Any]):
        """시스템 초기화 테스트"""
        print("\n🚀 시스템 초기화 테스트")
        print("-" * 40)
        
        if not models:
            self.test_result("시스템 초기화", False, "사용 가능한 모델 없음")
            return False
        
        # 첫 번째 사용 가능한 모델 선택
        model_key = list(models.keys())[0]
        print(f"   선택된 모델: {models[model_key]['name']}")
        
        try:
            start_time = time.time()
            result = await client.initialize_system(model_key, top_k=3)
            init_time = time.time() - start_time
            
            success = result.get('status') == 'success'
            self.test_result(
                "시스템 초기화",
                success,
                f"시간: {init_time:.2f}초, 메시지: {result.get('message', '')}"
            )
            return success
            
        except Exception as e:
            self.test_result("시스템 초기화", False, f"예외: {e}")
            return False
    
    async def test_system_status(self, client: RAGAPIClient):
        """시스템 상태 테스트"""
        print("\n📊 시스템 상태 테스트")
        print("-" * 40)
        
        try:
            status = await client.get_system_status()
            is_initialized = status.get('is_initialized', False)
            self.test_result(
                "시스템 상태",
                is_initialized,
                f"초기화됨: {is_initialized}, 모델: {status.get('model', 'None')}"
            )
            return status
            
        except Exception as e:
            self.test_result("시스템 상태", False, f"예외: {e}")
            return {}
    
    async def test_query_processing(self, client: RAGAPIClient):
        """질의 처리 테스트"""
        print("\n💬 질의 처리 테스트")
        print("-" * 40)
        
        test_queries = [
            "BC카드의 주요 서비스는 무엇인가요?",
            "신용카드 분실 시 어떻게 해야 하나요?",
            "카드 이용 안내 정보를 알려주세요."
        ]
        
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                start_time = time.time()
                response = await client.query(query, self.test_session_id)
                process_time = time.time() - start_time
                
                success = response.get('status') == 'success'
                if success:
                    successful_queries += 1
                    answer_preview = response.get('answer', '')[:100] + "..."
                    self.test_result(
                        f"질의 {i}",
                        True,
                        f"시간: {process_time:.2f}초, 답변: {answer_preview}"
                    )
                else:
                    self.test_result(
                        f"질의 {i}",
                        False,
                        f"오류: {response.get('message', 'Unknown error')}"
                    )
                    
            except Exception as e:
                self.test_result(f"질의 {i}", False, f"예외: {e}")
        
        return successful_queries
    
    async def test_chat_history(self, client: RAGAPIClient, expected_count: int):
        """대화 기록 테스트"""
        print("\n📝 대화 기록 테스트")
        print("-" * 40)
        
        try:
            # 대화 기록 조회
            history = await client.get_chat_history(self.test_session_id)
            actual_count = history.get('count', 0)
            
            self.test_result(
                "대화 기록 조회",
                actual_count == expected_count,
                f"예상: {expected_count}, 실제: {actual_count}"
            )
            
            # 대화 기록 삭제
            clear_result = await client.clear_chat_history(self.test_session_id)
            clear_success = clear_result.get('status') == 'success'
            
            self.test_result(
                "대화 기록 삭제",
                clear_success,
                f"메시지: {clear_result.get('message', '')}"
            )
            
            # 삭제 후 확인
            if clear_success:
                history_after = await client.get_chat_history(self.test_session_id)
                empty_history = history_after.get('count', 0) == 0
                
                self.test_result(
                    "삭제 후 확인",
                    empty_history,
                    f"삭제 후 대화 수: {history_after.get('count', 0)}"
                )
                
        except Exception as e:
            self.test_result("대화 기록 테스트", False, f"예외: {e}")
    
    async def test_session_management(self, client: RAGAPIClient):
        """세션 관리 테스트"""
        print("\n🔧 세션 관리 테스트")
        print("-" * 40)
        
        try:
            # 활성 세션 목록
            sessions = await client.get_active_sessions()
            session_count = sessions.get('count', 0)
            
            self.test_result(
                "활성 세션 조회",
                'sessions' in sessions,
                f"활성 세션 수: {session_count}"
            )
            
        except Exception as e:
            self.test_result("세션 관리 테스트", False, f"예외: {e}")
    
    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "="*50)
        print("🎯 테스트 결과 요약")
        print("="*50)
        print(f"전체 테스트: {self.total_tests}")
        print(f"성공: {self.passed_tests}")
        print(f"실패: {self.total_tests - self.passed_tests}")
        print(f"성공률: {(self.passed_tests / self.total_tests * 100):.1f}%" if self.total_tests > 0 else "0%")
        
        if self.passed_tests == self.total_tests:
            print("🎉 모든 테스트 통과!")
        else:
            print("⚠️ 일부 테스트 실패")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🧪 FastAPI RAG 시스템 종합 테스트")
        print("="*50)
        
        async with RAGAPIClient(self.base_url) as client:
            # 1. 기본 엔드포인트 테스트
            models = await self.test_basic_endpoints(client)
            
            # 2. 의존성 확인 테스트
            await self.test_dependency_check(client)
            
            # 3. 시스템 초기화 테스트
            init_success = await self.test_system_initialization(client, models)
            
            # 4. 시스템 상태 테스트
            await self.test_system_status(client)
            
            # 초기화가 성공한 경우에만 다음 테스트 진행
            if init_success:
                # 5. 질의 처리 테스트
                successful_queries = await self.test_query_processing(client)
                
                # 6. 대화 기록 테스트
                await self.test_chat_history(client, successful_queries)
                
                # 7. 세션 관리 테스트
                await self.test_session_management(client)
            else:
                print("\n⚠️ 초기화 실패로 인해 일부 테스트를 건너뜁니다.")
        
        # 결과 요약
        self.print_summary()


async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI RAG 시스템 테스트")
    parser.add_argument("--url", default="http://127.0.0.1:8110", help="API 서버 URL")
    
    args = parser.parse_args()
    
    tester = RAGAPITester(args.url)
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
