import os
import glob
import config

def load_prompt(filename):
    """从 prompts 文件夹读取 prompt 模板"""
    path = os.path.join(config.PROMPT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_report_files():
    """获取所有待处理的报告文件路径"""
    pattern = os.path.join(config.INPUT_DIR, "*.txt")
    return glob.glob(pattern)

def read_file_content(file_path):
    """读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_file_id(file_path):
    """从路径提取文件名作为 ID"""
    file_name = os.path.basename(file_path)
    return os.path.splitext(file_name)[0]
