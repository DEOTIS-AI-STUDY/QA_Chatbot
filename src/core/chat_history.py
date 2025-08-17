"""
대화 기록 관리 공통 모듈
- CLI와 Streamlit UI에서 공통으로 사용
- 일관된 대화 기록 관리 로직 제공
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatHistoryManager:
    """대화 기록 관리자 - CLI와 UI 공통 사용"""
    
    def __init__(self, max_history: int = 15):
        """
        Args:
            max_history: 최대 저장할 대화 개수
        """
        self.chat_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.context_window = 3  # 컨텍스트로 사용할 최근 대화 개수
    
    def add_chat(self, query: str, answer: str) -> None:
        """대화 기록 추가"""
        chat_entry = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        self.chat_history.append(chat_entry)
        
        # 최대 개수 제한
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
    
    def build_context_query(self, current_query: str) -> str:
        """대화 기록을 포함한 컨텍스트 질의 구성"""
        if not self.chat_history:
            return current_query
        
        # 최근 N개의 대화만 컨텍스트로 사용
        recent_history = self.chat_history[-self.context_window:]
        
        context_parts = []
        context_parts.append("이전 대화 내용:")
        
        for i, chat in enumerate(recent_history, 1):
            context_parts.append(f"Q{i}: {chat['query']}")
            # 답변은 200자로 제한하여 토큰 사용량 최적화
            answer_preview = chat['answer'][:200]
            if len(chat['answer']) > 200:
                answer_preview += "..."
            context_parts.append(f"A{i}: {answer_preview}")
        
        context_parts.append(f"\n현재 질문: {current_query}")
        context_parts.append("\n위의 이전 대화 내용을 참고하여 현재 질문에 답변해주세요.")
        
        return "\n".join(context_parts)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """대화 기록 반환"""
        return self.chat_history.copy()
    
    def get_history_count(self) -> int:
        """대화 기록 개수 반환"""
        return len(self.chat_history)
    
    def clear_history(self) -> None:
        """대화 기록 삭제"""
        self.chat_history.clear()
    
    def has_history(self) -> bool:
        """대화 기록 존재 여부"""
        return len(self.chat_history) > 0
    
    def get_recent_context_count(self) -> int:
        """최근 컨텍스트로 사용되는 대화 개수"""
        return min(self.context_window, len(self.chat_history))
    
    def get_summary_info(self) -> str:
        """대화 기록 요약 정보"""
        if not self.chat_history:
            return "대화 기록 없음"
        
        total_count = len(self.chat_history)
        context_count = self.get_recent_context_count()
        return f"{total_count}개 (최근 {context_count}개 참조 중)"


# CLI용 인터페이스 - 터미널 출력에 특화
class CLIChatHistoryInterface:
    """CLI 환경에서 대화 기록 표시/관리"""
    
    def __init__(self, manager: ChatHistoryManager):
        self.manager = manager
    
    def show_history(self) -> None:
        """CLI에서 대화 기록 표시"""
        if not self.manager.has_history():
            print("\n📝 저장된 대화 기록이 없습니다.")
            return
        
        history = self.manager.get_history()
        print(f"\n📝 대화 기록 ({len(history)}개):")
        print("=" * 60)
        
        for i, chat in enumerate(history, 1):
            print(f"\n[{i}] {chat['timestamp']}")
            print(f"Q: {chat['query']}")
            answer_preview = chat['answer'][:150]
            if len(chat['answer']) > 150:
                answer_preview += "..."
            print(f"A: {answer_preview}")
            print("-" * 40)
    
    def clear_history_with_confirmation(self) -> bool:
        """확인 후 대화 기록 삭제"""
        if not self.manager.has_history():
            print("\n📝 삭제할 대화 기록이 없습니다.")
            return False
        
        count = self.manager.get_history_count()
        confirm = input(f"\n📝 {count}개의 대화 기록을 삭제하시겠습니까? (y/N): ").strip().lower()
        
        if confirm in ['y', 'yes', '예']:
            self.manager.clear_history()
            print("✅ 대화 기록이 삭제되었습니다.")
            return True
        else:
            print("❌ 삭제가 취소되었습니다.")
            return False
    
    def show_status_info(self) -> None:
        """대화 상태 정보 표시"""
        if self.manager.has_history():
            summary = self.manager.get_summary_info()
            print(f"💭 대화 기록: {summary}")


# Streamlit용 인터페이스 - 웹 UI에 특화
class StreamlitChatHistoryInterface:
    """Streamlit 환경에서 대화 기록 표시/관리"""
    
    def __init__(self, manager: ChatHistoryManager):
        self.manager = manager
    
    def show_history(self) -> None:
        """Streamlit에서 대화 기록 표시"""
        import streamlit as st
        
        if not self.manager.has_history():
            st.info("저장된 대화 기록이 없습니다.")
            return
        
        history = self.manager.get_history()
        st.subheader(f"📝 대화 기록 ({len(history)}개)")
        
        # 최신 순으로 표시
        for i, chat in enumerate(reversed(history), 1):
            with st.expander(f"[{chat['timestamp']}] {chat['query'][:50]}..."):
                st.write("**질문:**")
                st.write(chat['query'])
                st.write("**답변:**")
                st.write(chat['answer'])
    
    def render_controls(self) -> tuple:
        """Streamlit 대화 기록 컨트롤 렌더링"""
        import streamlit as st
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        show_clicked = False
        clear_clicked = False
        
        with col1:
            if st.button("📝 대화기록 보기"):
                show_clicked = True
        
        with col2:
            if st.button("🗑️ 기록 삭제"):
                clear_clicked = True
                self.manager.clear_history()
                st.success("대화 기록이 삭제되었습니다.")
        
        with col3:
            if self.manager.has_history():
                summary = self.manager.get_summary_info()
                st.info(f"💭 대화 기록: {summary}")
        
        return show_clicked, clear_clicked
