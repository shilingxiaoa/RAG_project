import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIRECTORY, EMBEDDING_MODEL_PATH


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)


def split_texts(texts: list[str]) -> list[Document]:
    """将文本列表切分为文档块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = [Document(page_content=t) for t in texts]
    return splitter.split_documents(docs)


def load_or_build_vectorstore(texts: list[str], persist_dir: str):
    """如果 persist_dir 已有向量库则直接加载，否则构建并持久化。

    Args:
        texts: 文档文本列表（仅在首次构建时使用）
        persist_dir: 持久化目录路径

    Returns:
        Chroma 向量库
    """
    embedding = get_embeddings()
    chroma_db_path = os.path.join(persist_dir, "chroma.sqlite3")

    if os.path.exists(chroma_db_path):
        print(f"📂 加载已持久化的向量库: {persist_dir}")
        return Chroma(
            embedding_function=embedding,
            persist_directory=persist_dir,
        )

    # 首次构建
    splits = split_texts(texts)
    print(f"📄 切分为 {len(splits)} 个文本块，持久化到: {persist_dir}")
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_dir,
    )
