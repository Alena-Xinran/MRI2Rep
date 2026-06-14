import os
import json
import time
from litellm import completion_cost
import config

class AnalysisLogger:
    def __init__(self, file_id):
        # 路径: logs/{RUN_ID}/{FILE_ID}/
        self.base_path = os.path.join(config.LOG_BASE_DIR, config.RUN_ID, file_id)
        self.total_cost = 0.0
    
    def exists(self):
        """检查该 ID 的日志文件是否已存在"""
        final_json_file = os.path.join(self.base_path, "final_result.json")
        return os.path.exists(final_json_file)

    def setup(self):
        """创建日志目录"""
        os.makedirs(self.base_path, exist_ok=True)

    def log_step(self, step_name, prompt, response_obj, extracted_info=None):
        """记录单步详情到 txt"""
        content = response_obj.choices[0].message.content.strip() if response_obj else ""
        
        # 计算成本
        try:
            cost = completion_cost(completion_response=response_obj) if response_obj else 0.0
        except:
            cost = 0.0
        
        self.total_cost += cost
        usage = response_obj.usage if response_obj else None
        
        # 格式化日志内容
        log_text = (
            f"=== STEP: {step_name} ===\n\n"
            f"--- PROMPT ---\n{prompt}\n\n"
            f"--- RAW RESPONSE ---\n{content}\n\n"
            f"--- EXTRACTED INFO ---\n{extracted_info}\n\n"
            f"--- USAGE ---\n"
            f"Tokens: In={usage.prompt_tokens if usage else 0} / Out={usage.completion_tokens if usage else 0}\n"
            f"Cost: ${cost:.6f}\n"
        )
        
        filename = f"{step_name}.txt"
        with open(os.path.join(self.base_path, filename), "w", encoding="utf-8") as f:
            f.write(log_text)

    def save_final_result(self, data):
        """保存最终 JSON"""
        data["_meta"] = {
            "total_cost_usd": self.total_cost,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.MODEL_NAME
        }
        with open(os.path.join(self.base_path, "final_result.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

