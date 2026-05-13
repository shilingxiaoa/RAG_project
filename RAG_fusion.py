from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def build_fusion_chain(retriever, llm, fusion_llm=None):
    """
    构建RAG融合链（基于 RRfusion）。
    :param retriever: 向量检索器
    :param llm: 用于最终回答的LLM（需支持流式）
    :param fusion_llm: 用于生成扩展查询的LLM（可复用llm）
    :return: chain 对象，接受 {"question": question}，支持 .stream()
    """
    if fusion_llm is None:
        fusion_llm = llm

    # 查询生成链
    prompt_fusion = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询扩展专家。请从多个角度将用户问题改写为不同的检索查询，以提高召回率。

    要求：
    1. 生成 3-5 个不同的查询
    2. 每个查询应有不同侧重点：
       - 查询1：核心概念（What/什么是）
       - 查询2：实现方法（How/如何做）
       - 查询3：问题解决（Why/为什么/解决方案）
       - 查询4：最佳实践（Best practice/推荐做法）
       - 查询5：相关技术（Alternatives/相关工具）
    3. 每个查询 20-50 字
    4. 使用中文，保持问题原意

    输出格式（每行一个查询，不要编号）：
    [查询1]
    [查询2]
    [查询3]"""),

        ("human", """原始问题：{question}

    请生成多个检索查询：""")
    ])


    generate_queries = (
        prompt_fusion
        | fusion_llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    # 融合函数
    def reciprocal_rank_fusion(results: list[list], k=60):
        fused_scores = {}
        doc_map = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + k)
                doc_map[key] = doc
        sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    # 融合检索链
    fusion_retrieval = (
        generate_queries
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # 最终回答链
    final_prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"
    )
    final_chain = (
        {"context": fusion_retrieval, "question": itemgetter("question")}
        | final_prompt
        | llm
        | StrOutputParser()
    )
    return final_chain






