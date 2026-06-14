import csv
import os
import shutil

# ================= 配置 =================
INPUT_CSV = "liver_report.csv"      # 输入的 CSV 文件名
OUTPUT_BASE_DIR = "reports/all_splits"  # 输出的基础目录
BATCH_SIZE = 100                    # 每个文件夹存放的文件数量

def clean_report_text(text):
    """
    清理报告文本：
    1. 去除首尾的引号 (虽然 csv 模块会自动处理，但为了保险起见)
    2. 去除首尾空白
    3. 处理可能的转义字符
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # 如果 CSV 读取时遗留了引号，去除它们
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
        
    # 将 CSV 中常见的双引号转义 "" 替换为单引号 "
    text = text.replace('""', '"')
    
    return text

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误: 找不到输入文件 {INPUT_CSV}")
        return

    # 清空或创建输出目录（可选：如果想保留旧数据，注释掉 shutil.rmtree）
    if os.path.exists(OUTPUT_BASE_DIR):
        print(f"⚠️  警告: 输出目录 {OUTPUT_BASE_DIR} 已存在。")
    else:
        os.makedirs(OUTPUT_BASE_DIR)

    print(f"🚀 开始处理 {INPUT_CSV} ...")

    count = 0
    success_count = 0

    try:
        with open(INPUT_CSV, mode='r', encoding='utf-8', newline='') as csvfile:
            # 使用 csv.DictReader 自动处理 CSV 格式（包括引号包裹的多行文本）
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                count += 1
                
                report_id = row.get('id', '').strip()
                report_content = row.get('report', '')

                if not report_id:
                    print(f"⚠️  第 {count} 行缺少 ID，跳过。")
                    continue

                # 1. 计算批次文件夹名称 (1_100, 101_200, ...)
                # (count - 1) // 100 * 100 + 1 计算起始号
                batch_start = ((count - 1) // BATCH_SIZE) * BATCH_SIZE + 1
                batch_end = batch_start + BATCH_SIZE - 1
                batch_dir_name = f"{batch_start}_{batch_end}"
                
                target_dir = os.path.join(OUTPUT_BASE_DIR, batch_dir_name)
                os.makedirs(target_dir, exist_ok=True)

                # 2. 清理文本
                cleaned_content = clean_report_text(report_content)

                # 3. 写入文件
                file_path = os.path.join(target_dir, f"{report_id}.txt")
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    success_count += 1
                except Exception as e:
                    print(f"❌ 写入文件 {file_path} 失败: {e}")

    except Exception as e:
        print(f"❌ 读取 CSV 文件失败: {e}")
        return

    print(f"\n✅ 处理完成！")
    print(f"   - 总行数: {count}")
    print(f"   - 成功提取: {success_count}")
    print(f"   - 输出目录: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
