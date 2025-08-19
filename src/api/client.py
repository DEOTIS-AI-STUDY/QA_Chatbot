#!/usr/bin/env python3
"""
FastAPI RAG 시스템 클라이언트 예제
"""

import asyncio
import aiohttp
from typing import Dict, Any


class RAGAPIClient:
    """RAG API 클라이언트"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """의존성 확인"""
        async with self.session.get(f"{self.base_url}/dependencies") as response:
            return await response.json()
    
    async def get_available_models(self) -> Dict[str, Any]:
        """사용 가능한 모델 목록"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def initialize_system(self, model: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 시스템 초기화"""
        data = {
            "model": model,
            "top_k": top_k
        }
        async with self.session.post(
            f"{self.base_url}/initialize",
            json=data
        ) as response:
            return await response.json()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
    
    async def query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """질의 처리"""
        data = {
            "query": query,
            "session_id": session_id
        }
        async with self.session.post(
            f"{self.base_url}/query",
            json=data
        ) as response:
            return await response.json()
    
    async def get_chat_history(self, session_id: str = "default") -> Dict[str, Any]:
        """대화 기록 조회"""
        async with self.session.get(
            f"{self.base_url}/chat/history/{session_id}"
        ) as response:
            return await response.json()
    
    async def clear_chat_history(self, session_id: str = "default") -> Dict[str, Any]:
        """대화 기록 삭제"""
        async with self.session.delete(
            f"{self.base_url}/chat/history/{session_id}"
        ) as response:
            return await response.json()
    
    async def get_active_sessions(self) -> Dict[str, Any]:
        """활성 세션 목록"""
        async with self.session.get(f"{self.base_url}/sessions") as response:
            return await response.json()


async def example_usage():
    """사용 예제"""
    async with RAGAPIClient() as client:
        print("🔍 RAG API 클라이언트 테스트")
        
        # 1. 헬스 체크
        print("\n1. 헬스 체크")
        health = await client.health_check()
        print(f"   상태: {health}")
        
        # 2. 의존성 확인
        print("\n2. 의존성 확인")
        try:
            deps = await client.check_dependencies()
            print(f"   의존성: {deps['status']}")
            if deps['status'] == 'error':
                print(f"   문제점: {deps['issues']}")
        except Exception as e:
            print(f"   오류: {e}")
        
        # 3. 사용 가능한 모델 확인
        print("\n3. 사용 가능한 모델")
        models = await client.get_available_models()
        print(f"   모델: {list(models['models'].keys())}")
        
        # 4. 시스템 초기화
        print("\n4. 시스템 초기화")
        if models['models']:
            model_name = list(models['models'].keys())[0]
            print(f"   선택된 모델: {model_name}")
            
            try:
                init_result = await client.initialize_system(model_name, top_k=3)
                print(f"   초기화: {init_result['status']}")
                if init_result['status'] == 'success':
                    print(f"   초기화 시간: {init_result['initialization_time']:.2f}초")
                
                # 5. 시스템 상태 확인
                print("\n5. 시스템 상태")
                status = await client.get_system_status()
                print(f"   초기화됨: {status['is_initialized']}")
                print(f"   모델: {status['model']}")
                
                # 6. 질의 처리
                print("\n6. 질의 처리")
                test_queries = [
                    "BC카드의 주요 서비스는 무엇인가요?",
                    "신용카드 분실 시 어떻게 해야 하나요?"
                ]
                
                for query in test_queries:
                    print(f"   질문: {query}")
                    try:
                        response = await client.query(query, "test_session")
                        if response['status'] == 'success':
                            print(f"   답변: {response['answer'][:100]}...")
                            print(f"   처리 시간: {response['processing_time']:.2f}초")
                        else:
                            print(f"   오류: {response['message']}")
                    except Exception as e:
                        print(f"   질의 오류: {e}")
                
                # 7. 대화 기록 확인
                print("\n7. 대화 기록")
                history = await client.get_chat_history("test_session")
                print(f"   대화 개수: {history['count']}")
                
            except Exception as e:
                print(f"   초기화 오류: {e}")
        else:
            print("   사용 가능한 모델이 없습니다.")


def sync_example():
    """동기 버전 예제"""
    import requests
    
    base_url = "http://127.0.0.1:8080"
    
    print("🔍 RAG API 동기 클라이언트 테스트")
    
    # 헬스 체크
    response = requests.get(f"{base_url}/health")
    print(f"헬스 체크: {response.json()}")
    
    # 사용 가능한 모델
    response = requests.get(f"{base_url}/models")
    models = response.json()
    print(f"사용 가능한 모델: {list(models['models'].keys())}")
    
    # 시스템 초기화 (첫 번째 모델 사용)
    if models['models']:
        model_name = list(models['models'].keys())[0]
        init_data = {"model": model_name, "top_k": 3}
        response = requests.post(f"{base_url}/initialize", json=init_data)
        init_result = response.json()
        print(f"초기화: {init_result}")
        
        if init_result.get('status') == 'success':
            # 질의 처리
            query_data = {
                "query": "BC카드의 주요 서비스는 무엇인가요?",
                "session_id": "sync_test"
            }
            response = requests.post(f"{base_url}/query", json=query_data)
            result = response.json()
            print(f"질의 결과: {result}")


if __name__ == "__main__":
    print("FastAPI RAG 클라이언트 예제")
    print("=" * 50)
    
    # 비동기 예제 실행
    asyncio.run(example_usage())
    
    print("\n" + "=" * 50)
    print("동기 클라이언트 예제")
    
    # 동기 예제 실행
    try:
        import requests
        sync_example()
    except ImportError:
        print("❌ requests 라이브러리가 필요합니다: pip install requests")
    except Exception as e:
        if "requests" in str(type(e)):
            print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        else:
            print(f"❌ 오류: {e}")
