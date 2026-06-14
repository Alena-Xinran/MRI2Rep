import sys
import os
import utils.io as utils_io
from utils.logger import AnalysisLogger
import pipeline
import config  # 必须导入配置以获取路径

# ================= 辅助类：同时输出到控制台和文件 =================
class LoggerWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8", buffering=1) # buffering=1 (行缓冲)

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() # 确保实时写入文件，防止程序崩溃时丢失日志
        except Exception:
            pass # 防止日志写入出错导致主程序崩溃

    def flush(self):
        # 必须实现 flush 方法，因为 print 等函数会调用它
        self.terminal.flush()
        self.log.flush()

# ================= 主程序 =================

def process_single_case(file_path):
    file_id = utils_io.get_file_id(file_path)
    
    # 1. 初始化日志
    logger = AnalysisLogger(file_id)
    
    # 2. 检查断点续传
    if logger.exists():
        print(f"⏩ Skipping {file_id}: Already processed.")
        return

    print(f"\n🚀 Processing ID: {file_id}")
    logger.setup() # 创建文件夹
    
    try:
        # 3. 读取报告
        report_text = utils_io.read_file_content(file_path)
        
        # 4. 执行 Pipeline
        # Step 1
        organ_status = pipeline.run_step_1_organ(report_text, logger)
        
        # Step 2
        final_organ_status = pipeline.run_step_2_verification(report_text, organ_status, logger)
        
        # Step 3
        raw_tumors, consistent, stats = pipeline.run_step_3_tumor_extraction(report_text, logger)
        
        # 5. 数据组装
        structured_tumors = pipeline.parse_tumor_records(raw_tumors)
        
        final_data = {
            "id": file_id,
            "organ_status": final_organ_status,
            "tumors": structured_tumors,
            "consistent": consistent,
            "consistency_details": stats
        }
        
        # 6. 保存
        logger.save_final_result(final_data)
        print(f"✅ Finished {file_id}. Cost: ${logger.total_cost:.6f}")
        
    except Exception as e:
        # 使用 sys.stderr 输出，也会被捕获到 log 文件中
        print(f"❌ Critical Error processing {file_id}: {e}", file=sys.stderr)

def main():
    # === 设置全局日志捕获 ===
    # 1. 确定本次运行的根目录 (logs/batch_run_vX)
    run_dir = os.path.join(config.LOG_BASE_DIR, config.RUN_ID)
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. 定义全量日志文件路径
    console_log_path = os.path.join(run_dir, "all_console_log.txt")
    
    # 3. 重定向 stdout 和 stderr
    # 保存原始的 stdout 以便后续（如果需要）恢复，但在脚本中通常不需要
    original_stdout = sys.stdout
    dual_writer = LoggerWriter(console_log_path)
    
    sys.stdout = dual_writer
    sys.stderr = dual_writer # 把错误信息也写进去
    
    print(f"=== Starting Batch Run: {config.RUN_ID} ===")
    print(f"Logs are being saved to: {console_log_path}")
    
    # === 开始业务逻辑 ===
    files = utils_io.get_report_files()
    
    if not files:
        print("No files found in reports directory.")
        return
        
    print(f"Found {len(files)} files. Starting pipeline...")
    
    for file_path in files:
        process_single_case(file_path)
        
    print(f"\n=== Batch Run Completed ===")

if __name__ == "__main__":
    main()
