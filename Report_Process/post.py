import os
import json
import csv
import glob

# ================= 配置 =================
split_dirname = os.getenv("SPLIT_DIRNAME", "1_100")  # 每次运行可以改这个
INPUT_DIR = f"./logs/batch_run_{split_dirname}" 
OUTPUT_CSV = f"{INPUT_DIR}/Summary.csv"

# ================= 排序优先级定义 =================
TYPE_PRIORITY = {
    "APHE_WO": 1,    # 确诊 HCC (LR-5)
    "APHE_NoWO": 2,  # 可疑/灌注异常
    "RIM_ATYP": 3,   # 恶性/不典型/治疗痕迹
    "HEM": 4,        # 血管瘤
    "CYST": 5,       # 囊肿
    "FAT": 6,        # 脂肪变性
    "OTHER": 99,
    "Unknown": 100
}

POS_PRIORITY = {
    "CAUDATE": 1,    # Seg 1
    "L_LAT": 2,      # Seg 2, 3
    "L_MED": 3,      # Seg 4
    "R_ANT": 4,      # Seg 5, 8
    "R_POST": 5,     # Seg 6, 7
    "R_JUNCTION": 6, 
    "L_DIFFUSE": 7,
    "R_DIFFUSE": 8,
    "DIFFUSE": 9,
    "Unknown": 99
}

def calculate_merged_quantity(quantities):
    total = 0
    for q in quantities:
        q_str = str(q).strip()
        if "multiple" in q_str.lower():
            return "Multiple"
        try:
            num = int(q_str)
            total += num
        except ValueError:
            return "Multiple"
    return str(total)

def merge_tumors(tumor_list):
    if not tumor_list:
        return []

    grouped = {}
    for t in tumor_list:
        t_type = t.get("type", "Unknown").strip()
        t_pos = t.get("position", "Unknown").strip()
        t_qty = t.get("quantity", "1").strip()
        
        key = (t_type, t_pos)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(t_qty)
    
    merged_list = []
    for (t_type, t_pos), qty_list in grouped.items():
        final_qty = calculate_merged_quantity(qty_list)
        merged_list.append({
            "type": t_type,
            "position": t_pos,
            "quantity": final_qty
        })
    
    # 排序：先类型，后位置
    merged_list.sort(key=lambda x: (
        TYPE_PRIORITY.get(x['type'], 100),
        POS_PRIORITY.get(x['position'], 100)
    ))
    return merged_list

def format_sequence(organ_status, tumors):
    parts = [organ_status]
    for t in tumors:
        parts.append(t["type"])
        parts.append(t["position"])
        parts.append(t["quantity"])
    return ", ".join(parts)

def process_batch(input_dir, output_file):
    search_pattern = os.path.join(input_dir, "*", "final_result.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"❌ 在 {input_dir} 下未找到任何 final_result.json 文件。")
        return

    print(f"🔍 找到 {len(files)} 个结果文件，开始处理...")
    
    csv_rows = []
    grand_total_cost = 0.0  # 总花费累加器

    yes_rate = 0.0    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            case_id = data.get("id", "Unknown")
            is_consistent = "Yes" if data.get("consistent", False) else "No"
            organ_status = data.get("organ_status", "Unknown")
            raw_tumors = data.get("tumors", [])
            
            # === 获取花费 ===
            # 从 _meta 中提取，如果没有则为 0
            cost = data.get("_meta", {}).get("total_cost_usd", 0.0)
            grand_total_cost += cost
            
            merged_tumors = merge_tumors(raw_tumors)
            sequence_str = format_sequence(organ_status, merged_tumors)

            yes_rate += (is_consistent == 'Yes')
            
            csv_rows.append({
                "id": case_id,
                "consistency": is_consistent,
                "cost": f"{cost:.6f}", # 保留6位小数
                "sequence": sequence_str
            })
            
        except Exception as e:
            print(f"⚠️ 处理文件 {file_path} 时出错: {e}")

    # 按 ID 排序
    csv_rows.sort(key=lambda x: x["id"])

    # 更新 Header
    headers = ["id", "consistency", "cost", "sequence"]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Yes Rate: {yes_rate / len(files)}")
        print(f"✅ 处理完成！结果已保存至: {output_file}")
        print(f"💰 本次批量处理总花费: ${grand_total_cost:.4f}")
        
    except Exception as e:
        print(f"❌ 写入 CSV 失败: {e}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在 {INPUT_DIR}")
    else:
        process_batch(INPUT_DIR, OUTPUT_CSV)
