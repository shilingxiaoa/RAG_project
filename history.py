from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List

class LimitedChatHistory:
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages
        self._history = InMemoryChatMessageHistory()

    def add_messages(self, message):
        self._history.add_message(message)
        messages = self._history.messages
        if len(messages) > self.max_messages:
            self._history.messages = messages[-self.max_messages:]

    @property
    def messages(self) -> List:
        return self._history.messages

    @messages.setter
    def messages(self, value):
        self._history.messages = value

    def clear(self):
        self._history.clear()

# 全局历史存储
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = LimitedChatHistory(max_messages=5)
    return store[session_id]