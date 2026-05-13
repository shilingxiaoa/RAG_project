from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)
from datasets import Dataset, load_dataset as hf_load_dataset
from prompts import get_prompt
from indexing import load_or_build_vectorstore
from config import PERSIST_DIRECTORY
import json
import csv
import os
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========== 数据集加载 ==========

def load_evaluation_dataset(source: str, format: str = "auto", **kwargs):
    """加载评估数据集，支持多种来源。

    Args:
        source: 文件路径（csv/json/jsonl）或 HuggingFace 数据集名称
        format: "csv", "json", "jsonl", "hf", 或 "auto"（从扩展名推断）
        **kwargs: 额外参数（如 split、config 等）

    Returns:
        HuggingFace Dataset
    """
    if format == "auto":
        ext = os.path.splitext(source)[1].lower()
        format_map = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}
        format = format_map.get(ext, "hf")

    if format == "csv":
        return _load_from_csv(source, **kwargs)
    elif format == "json":
        return _load_from_json(source, **kwargs)
    elif format == "jsonl":
        return _load_from_jsonl(source, **kwargs)
    elif format == "hf":
        return _load_from_huggingface(source, **kwargs)
    else:
        raise ValueError(f"不支持的数据集格式: {format}")


def _load_from_csv(path: str, **kwargs) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return Dataset.from_list(rows)


def _load_from_json(path: str, **kwargs) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def _load_from_jsonl(path: str, **kwargs) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _load_from_huggingface(source: str, **kwargs) -> Dataset:
    config_name = kwargs.pop("config", None)
    split = kwargs.pop("split", "test")
    return hf_load_dataset(source, config_name, split=split, **kwargs)


# RAGAS 兼容的数据集预设
DOMAIN_DATASETS = {
    "金融 (FiQA)": {
        "source": "explodinggradients/fiqa",
        "config": "ragas_eval",
        "split": "baseline",
    },
    "人权 (Amnesty)": {
        "source": "explodinggradients/amnesty_qa",
        "config": "english_v2",
        "split": "baseline",
    },
}


def _sanitize_dirname(name: str) -> str:
    """将名称转为安全的目录名。"""
    return re.sub(r'[^\w一-鿿]+', '_', name).strip('_')


# 为预设数据集生成持久化目录路径
for _cfg in DOMAIN_DATASETS.values():
    raw = _cfg["source"].replace("/", "_")
    if _cfg.get("config"):
        raw += f"_{_cfg['config']}"
    _cfg["persist_dir"] = os.path.join(PERSIST_DIRECTORY, _sanitize_dirname(raw))


def _resolve_ground_truth_column(dataset: Dataset) -> str | None:
    """兼容 ground_truth / ground_truths 两种列名。"""
    for col in ("ground_truth", "ground_truths"):
        if col in dataset.column_names:
            return col
    return None


# ========== 从数据集构建向量库 ==========

def build_retriever_from_dataset(dataset, embeddings, persist_dir=None):
    """从数据集的 contexts 列构建或加载向量检索器。

    如果 persist_dir 已有持久化的向量库则直接加载，否则构建并持久化。

    Args:
        dataset: HuggingFace Dataset，需包含 "contexts" 列
        embeddings: 用于向量化的 embedding 模型
        persist_dir: 持久化目录，不传则不持久化（每次重新构建）

    Returns:
        Chroma 向量库的 retriever
    """
    if "contexts" not in dataset.column_names:
        raise ValueError("数据集中缺少 'contexts' 列，无法构建知识库")

    # 提取所有上下文并去重
    all_contexts = set()
    for ctx_list in dataset["contexts"]:
        for ctx in ctx_list:
            all_contexts.add(ctx.strip())

    print(f"📚 从数据集提取了 {len(all_contexts)} 个唯一文档块")

    if persist_dir:
        vectorstore = load_or_build_vectorstore(list(all_contexts), persist_dir)
    else:
        # 不持久化，直接用内存模式构建
        from indexing import split_texts
        from langchain_community.vectorstores import Chroma
        splits = split_texts(list(all_contexts))
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=None,
        )

    return vectorstore.as_retriever()


# ========== 单数据集评估 ==========

def run_ragas_evaluation(
    dataset,
    retriever,
    llm,
    embeddings,
    eval_llm=None,
    question_column="question",
):
    """对给定的数据集运行 RAGAS 评估。

    Args:
        dataset: HuggingFace Dataset，至少包含 question_column 列
        retriever: 检索器对象（.invoke() 方法）
        llm: 用于生成答案的 LLM
        embeddings: 用于评估的 embedding 模型
        eval_llm: 用于 RAGAS 打分的 LLM（默认同 llm）
        question_column: 问题列的列名

    Returns:
        (result, eval_dataset) — ragas 评估结果 + 构造的评估数据集
    """
    questions = dataset[question_column]
    gt_col = _resolve_ground_truth_column(dataset)
    has_ground_truth = gt_col is not None

    print(f"\n{'='*50}")
    print(f"  样本数: {len(questions)}")
    print(f"  含标准答案: {has_ground_truth}")
    print(f"{'='*50}")

    answers = []
    contexts = []

    for i, q in enumerate(questions):
        print(f"   [{i+1}/{len(questions)}] 检索+生成中...", end=" \r")

        # 1. 检索
        docs = retriever.invoke(q)
        context_strs = [doc.page_content for doc in docs]
        contexts.append(context_strs)

        # 2. 生成答案
        prompt = get_prompt()
        context_str = "\n\n---\n\n".join(context_strs)
        messages = prompt.invoke({
            "context": context_str,
            "history": [],
            "question": q,
        })
        response = llm.invoke(messages.to_messages())
        answers.append(response.content)

    print()

    # 3. 构造评估数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if has_ground_truth:
        raw = dataset[gt_col]
        # RAGAS 要求 ground_truth 是字符串，数据集里可能是 ["text"] 格式
        data["ground_truth"] = [
            gt[0] if isinstance(gt, list) and len(gt) > 0 else (gt or "")
            for gt in raw
        ]

    eval_dataset = Dataset.from_dict(data)

    # 4. 选择指标
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics.extend([answer_correctness, context_recall])

    # 5. 运行评估
    print("⏳ 运行 RAGAS 评估...")
    evaluator_llm = eval_llm or llm
    result = evaluate(eval_dataset, metrics=metrics, llm=evaluator_llm, embeddings=embeddings)

    # 6. 输出结果
    df = result.to_pandas()
    df.insert(0, "question", questions)
    display_cols = ["question", "faithfulness", "answer_relevancy"]
    if has_ground_truth:
        display_cols.extend(["answer_correctness", "context_recall"])
    print(df[display_cols].to_string(index=False))

    avg = df[[c for c in display_cols if c != "question"]].mean()
    print("\n📊 平均得分：")
    print(avg)

    return result, eval_dataset


# ========== 多数据集比较评估 ==========

def run_comparative_evaluation(
    datasets: dict[str, Dataset],
    llm,
    embeddings,
    eval_llm=None,
):
    """在多个领域数据集上构建向量库并运行评估，生成对比报告。

    对每个领域：
      1. 从数据集的 contexts 中构建该领域的向量知识库
      2. 用该知识库检索并生成答案
      3. 用 RAGAS 评估

    Args:
        datasets: {领域名称: HuggingFace Dataset}
        llm: 用于生成答案的 LLM
        embeddings: 用于评估的 embedding 模型
        eval_llm: 用于 RAGAS 打分的 LLM（默认同 llm）

    Returns:
        {领域名称: (result, eval_dataset, retriever)}
    """
    all_results = {}

    for domain_name, dataset in datasets.items():
        print(f"\n{'#'*60}")
        print(f"#  领域: {domain_name}")
        print(f"{'#'*60}")

        # 为该领域构建或加载向量库
        # 预设数据集有自己的 persist_dir，自定义数据集不持久化
        cfg = DOMAIN_DATASETS.get(domain_name)
        persist_dir = cfg.get("persist_dir") if cfg else None
        retriever = build_retriever_from_dataset(dataset, embeddings, persist_dir)

        result, eval_ds = run_ragas_evaluation(
            dataset, retriever, llm, embeddings, eval_llm=eval_llm,
        )
        all_results[domain_name] = (result, eval_ds, retriever)

    # ===== 生成汇总对比表 =====
    print("\n" + "="*60)
    print("📊 多领域评估汇总对比")
    print("="*60)

    rows = []
    for domain_name, (result, _, _) in all_results.items():
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c != "question"]
        avg_row = df[metric_cols].mean(numeric_only=True)
        rows.append({"领域": domain_name, **avg_row.to_dict()})

    import pandas as pd
    summary = pd.DataFrame(rows).set_index("领域")
    print(summary.to_string())
    print("="*60)

    return all_results


# ========== 加载多领域数据集 ==========

def load_domain_datasets(domain_names: list[str] | None = None):
    """加载预设的多领域评估数据集。

    Args:
        domain_names: 领域名称列表，默认加载全部

    Returns:
        {领域名称: Dataset}
    """
    if domain_names is None:
        domain_names = list(DOMAIN_DATASETS.keys())

    datasets = {}
    for name in domain_names:
        if name not in DOMAIN_DATASETS:
            print(f"⚠️ 未知领域: {name}，跳过")
            continue
        cfg = DOMAIN_DATASETS[name]
        print(f"📥 加载数据集: {name} ({cfg['source']}, {cfg['config']})")
        ds = load_evaluation_dataset(
            cfg["source"],
            format="hf",
            config=cfg["config"],
            split=cfg["split"],
        )
        datasets[name] = ds
        print(f"   ✅ {len(ds)} 条样本")

    return datasets
