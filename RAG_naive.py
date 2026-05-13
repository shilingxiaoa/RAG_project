from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from prompts import get_prompt
from history import get_session_history, store

def build_naive_chain(retriever, llm):
    prompt = get_prompt()

    # 基本链（不带历史包装）
    base_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 包装历史
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    return chain_with_history

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])