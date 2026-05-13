from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def build_hyde_chain(retriever, llm, hyde_llm=None):
    """
    构建HyDE链。
    :param retriever: 向量检索器
    :param llm: 最终回答的LLM（需支持流式）
    :param hyde_llm: 用于生成假设文档的LLM（可复用llm）
    :return: chain 对象，接受 {"question": question}，支持 .stream()
    """
    if hyde_llm is None:
        hyde_llm = llm

    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的技术文档撰写助手。你的任务是基于用户的问题，生成一段**假设性的答案文档**。

    要求：
    1. 语言：中文
    2. 风格：专业、详细、结构化，像是从官方技术文档或学术论文中摘录的段落
    3. 长度：200-400字
    4. 内容：即使不确定具体细节，也要根据常识生成合理、连贯的答案
    5. 格式：使用清晰的段落，必要时包含技术术语和具体步骤

    注意：这是用于检索的假设性文档，目的是在向量空间中匹配到真实的相关文档。"""),

        ("human", """问题：{question}

    请生成一段假设性的技术文档来回答上述问题：""")
    ])


    generate_hyde = hyde_prompt | hyde_llm | StrOutputParser()

    # 使用假设文档进行检索
    def hyde_retrieve(question: str):
        hypothetical_doc = generate_hyde.invoke({"question": question})
        docs = retriever.invoke(hypothetical_doc)
        return "\n\n".join([doc.page_content for doc in docs])

    # 最终回答链
    final_prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
    )
    final_chain = (
        {"context": hyde_retrieve, "question": itemgetter("question")}
        | final_prompt
        | llm
        | StrOutputParser()
    )
    return final_chain

