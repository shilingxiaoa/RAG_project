from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TIMEOUT,
    ZHIPU_MODEL, ZHIPU_BASE_URL, ZHIPU_API_KEY,
)


def get_llm():
    if not LLM_API_KEY:
        raise ValueError("API key 未设置！")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        streaming=True,
    )


def get_non_stream_llm():(非流式输出模型)
    if not LLM_API_KEY:
        raise ValueError("API key 未设置！")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
    )


def get_eval_llm():
    """RAGAS 评估用的 LLM（使用OLLAMA免费模型，不消耗 DeepSeek API token）。"""
    return ChatOllama(
        model="qwen2.5:7b",  
        temperature=0.1,  # 降低随机性，处理 RAG 评估更稳定
        num_predict=512,  # 限制输出长度
        top_k=1  # 完全确定性输出，有助于避免解析错误
    )
