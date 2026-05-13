from evaluation import load_domain_datasets, DOMAIN_DATASETS, build_retriever_from_dataset
from llm_setup import get_llm, get_eval_llm
from RAG_naive import build_naive_chain
from RAG_fusion import build_fusion_chain
from RAG_HyDE import build_hyde_chain
from validation import validate_input
from evaluation import run_comparative_evaluation


def stream_response(chain, inputs, config=None):
    for chunk in chain.stream(inputs, config=config):
        print(chunk, end="", flush=True)


def select_dataset_for_chat(embeddings):
    """启动时让用户选择用作聊天知识库的数据集。"""
    print("\n📚 选择聊天知识库")
    print("可用的领域数据集：")
    names = list(DOMAIN_DATASETS.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

    choice = input("\n请选择（输入编号，留空默认选第一个）: ").strip()
    if choice == "":
        choice = "1"

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            selected_name = names[idx]
        else:
            selected_name = names[0]
    else:
        selected_name = names[0]

    print(f"📥 加载数据集: {selected_name}")
    datasets = load_domain_datasets([selected_name])
    dataset = datasets[selected_name]
    cfg = DOMAIN_DATASETS.get(selected_name)
    persist_dir = cfg.get("persist_dir") if cfg else None
    retriever = build_retriever_from_dataset(dataset, embeddings, persist_dir)

    return retriever, selected_name


def main():
    base_llm = get_llm()
    eval_llm = get_eval_llm()

    # 0. 初始化 embedding 模型
    from indexing import get_embeddings
    embeddings = get_embeddings()

    # 1. 选择聊天知识库
    retriever, current_domain = select_dataset_for_chat(embeddings)

    # 2. 构建各模式链
    current_mode = "naive"
    current_chain = build_naive_chain(retriever, base_llm)
    mode_mapping = {
        "naive": lambda: build_naive_chain(retriever, base_llm),
        "fusion": lambda: build_fusion_chain(retriever, base_llm),
        "hyde": lambda: build_hyde_chain(retriever, base_llm),
    }

    print(f"\n✅ 知识库: {current_domain}  |  模式: {current_mode.upper()}")
    print("输入 '评估' 运行 RAGAS 评测，输入 'exit' 退出")


    user_name=input("请输入用户名：")
    while True:
        user_input = input(f"\n请输入问题（'模式 naive/fusion/hyde' 切换，'评估' 运行RAGAS，'exit' 结束）: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("再见！")
            break

        # 模式切换
        if user_input.lower().startswith("模式 "):
            mode = user_input.split(" ")[1].lower()
            if mode in mode_mapping:
                current_mode = mode
                current_chain = mode_mapping[mode]()
                print(f"🔄 已切换至 {mode.upper()} 模式")
                from history import store
                store.clear()
                continue
            else:
                print(f"❌ 未知模式 {mode}，可选: naive, fusion, hyde")
                continue

        # RAGAS 评估
        if user_input.lower() in ["评估", "eval", "evaluate"]:
            print("\n📋 选择评估数据集：")
            names = list(DOMAIN_DATASETS.keys())
            for i, name in enumerate(names, 1):
                print(f"   {i}. {name}")
            print(f"   {len(names)+1}. 自定义文件路径")
            choice = input("\n请选择（编号，逗号分隔多选，留空全选）: ").strip()

            if choice == "":
                datasets = load_domain_datasets()
            elif choice.isdigit() and int(choice) == len(names) + 1:
                ds_path = input("请输入数据集路径（CSV/JSON/JSONL）: ").strip()
                from evaluation import load_evaluation_dataset, run_ragas_evaluation
                eval_dataset = load_evaluation_dataset(ds_path)
                run_ragas_evaluation(eval_dataset, retriever, base_llm, embeddings, eval_llm=eval_llm)
                continue
            else:
                selected = []
                for part in choice.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(names):
                            selected.append(names[idx])
                datasets = load_domain_datasets(selected)

            run_comparative_evaluation(datasets, base_llm, embeddings, eval_llm=eval_llm)
            continue

        # 输入验证
        is_valid, cleaned_input = validate_input(user_input)
        if not is_valid:
            print(f"⚠️ {cleaned_input}")
            continue

        print(f"🤖 [{current_mode.upper()}模式] 生成答案中...")
        print("=" * 50)
        print("助手：", end="", flush=True)
        try:
            if current_mode == "naive":
                stream_response(
                    current_chain,
                    {"question": cleaned_input},
                    {"configurable": {"session_id": f"session_{user_name}"}}
                )
            else:
                stream_response(
                    current_chain,
                    {"question": cleaned_input}
                )
        except Exception as e:
            print(f"\n❌ 错误：{e}")
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
