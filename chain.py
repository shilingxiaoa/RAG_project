from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from prompts import get_prompt
from indexing import build_vectorstore
from llm_setup import get_llm
from history import get_session_history

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def build_rag_chain():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    prompt = get_prompt()

    # 基本链
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
    conversational_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    return conversational_chain, retriever