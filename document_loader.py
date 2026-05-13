from typing import Union
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import requests
from tempfile import NamedTemporaryFile
import os
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.documents import Document




def load_document(source:Union[str,Path])->list[Document]:
    if isinstance(source,str):
        if source.startswith("http"):
            return load_from_url(source)
        elif is_file_path(source):
            return load_from_file(source)
        else:
            return load_from_text(source)
    elif isinstance(source,Path):
        return load_from_Path(source)


def load_from_url(url: str) -> list[Document]:
    # 判断 URL 是否指向纯文本文件（根据后缀或 Content-Type）
    if _is_plain_text_url(url):
        return _load_from_url_as_text(url)

    # 否则按 HTML 加载
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={"features": "html.parser"}
    )
    # 设置 user-agent 避免警告
    loader.requests_kwargs = {
        "headers": {"User-Agent": "Mozilla/5.0 (compatible; RAGEvaluation/1.0)"}
    }
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["type"] = "web"
    return docs


def _is_plain_text_url(url: str) -> bool:
    """判断 URL 是否指向纯文本文件（简单后缀匹配，也可尝试 HEAD 请求）"""
    # 常见纯文本后缀
    plain_extensions = ('.txt', '.md', '.csv', '.json', '.xml')
    if url.lower().endswith(plain_extensions):
        return True
    # 若后缀不明确，可通过 HEAD 请求 Content-Type 判断（但为避免额外网络请求，先只用后缀）
    return False


def _load_from_url_as_text(url: str) -> list[Document]:
    """通过 requests 下载纯文本并用 TextLoader 加载（避免 BeautifulSoup 解析）"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGEvaluation/1.0)"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    # 写入临时文件
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(response.text)
        temp_path = f.name
    try:
        loader = TextLoader(temp_path, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = url
            doc.metadata["type"] = "web_text"
        return docs
    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)



def is_file_path(path:str)->bool:
    return Path(path).is_file()

def load_from_file(file_path:str)->list[Document]:
    path=Path(file_path)
    ext=path.suffix.lower()

    if ext == ".pdf":
        loader=PyPDFLoader(file_path)
        docs=loader.load()
    elif ext == ".txt":
        loader=TextLoader(file_path,encoding="utf-8")
        docs=loader.load()
    elif ext in [".md", ".markdown"]:
        loader=TextLoader(file_path,encoding="utf-8")
        docs=loader.load
    elif ext in [".doc", ".docx"]:
        loader=Docx2txtLoader(file_path)
        docs=loader.load()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    for doc in docs:
        doc.metadata["source"]=file_path
        doc.metadata["type"]="file"
        doc.metadata["file_name"]=path.name
    return docs
def load_from_text(text:str)->list[Document]:
    doc=Document(
        page_content=text,
        metadata={
            "source": "text",
            "type": "text",
            "length": len(text)
        }
    )
    return [doc]
def load_from_Path(path:Path)->list[Document]:
    return load_from_file(str(path))





