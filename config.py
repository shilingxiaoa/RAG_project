import os

# Embedding 模型路径(如果需要)
EMBEDDING_MODEL_PATH = "YOUR_EMBEDDING_MODEL_PATH"

# 文本分块参数（加载自定义文档时使用）
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# 向量库持久化路径
PERSIST_DIRECTORY = "./chroma_db"

# 对话生成 LLM 配置（DeepSeek API）
LLM_MODEL = "YOUR_MODEL
LLM_API_KEY = "YOUR_API_KEY"
LLM_BASE_URL = "YOUR_MODEL_BASE_URL"("https://api.deepseek.com")
LLM_TIMEOUT = 30.0

# RAGAS 评估 LLM 配置（智谱 GLM 免费模型）
ZHIPU_MODEL = "glm-4.7-flash"
ZHIPU_API_KEY = "YOUR_API_KEY"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 对话历史限制
MAX_HISTORY_MESSAGES = 5

# 离线环境变量
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
