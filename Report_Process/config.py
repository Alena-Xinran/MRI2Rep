import os

# API 设置
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-你的key")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek/deepseek-chat"

# 路径设置
split_dirname = os.getenv("SPLIT_DIRNAME", "1_100")  # 每次运行可以改这个
INPUT_DIR = f"./reports/all_splits/{split_dirname}"
LOG_BASE_DIR = "./logs"
PROMPT_DIR = "./prompts"

# 运行参数
RUN_ID = f"batch_run_{INPUT_DIR.split('/')[-1]}" 
CONSISTENCY_ROUNDS = 3   # Step 3 重复次数

# 自动创建必要目录
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(LOG_BASE_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)
