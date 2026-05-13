"""一次性下载所有评估数据集到本地，后续加载直接从本地读取"""
import os
import sys
from datasets import load_dataset

# 使用国内镜像加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 本地缓存目录（项目内）
LOCAL_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_datasets")

DATASETS_TO_DOWNLOAD = {
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


def download_all():
    print("=" * 60)
    print("开始下载评估数据集到本地 ...")
    print(f"本地缓存目录: {LOCAL_DATASETS_DIR}")
    print("=" * 60)

    for name, cfg in DATASETS_TO_DOWNLOAD.items():
        save_dir = os.path.join(LOCAL_DATASETS_DIR, name)
        if os.path.exists(os.path.join(save_dir, "dataset_dict.json")):
            print(f"✅ [{name}] 已存在，跳过下载")
            continue

        print(f"\n📥 正在下载 [{name}] ({cfg['source']}, {cfg['config']}, {cfg['split']}) ...")
        print(f"   这可能花费几分钟，请耐心等待 ...")

        try:
            dataset = load_dataset(
                cfg["source"],
                cfg["config"],
                split=cfg["split"],
                trust_remote_code=True,
            )
            print(f"   下载完成！共 {len(dataset)} 条样本")

            # 保存到本地
            os.makedirs(save_dir, exist_ok=True)
            dataset.save_to_disk(save_dir)
            print(f"   ✅ 已保存到: {save_dir}")

        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            continue

    print("\n" + "=" * 60)
    print("全部完成！本地数据集位置:")
    for name in DATASETS_TO_DOWNLOAD:
        path = os.path.join(LOCAL_DATASETS_DIR, name)
        exists = os.path.exists(os.path.join(path, "dataset_dict.json"))
        status = "✅ 已下载" if exists else "❌ 未下载"
        print(f"  {status}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
