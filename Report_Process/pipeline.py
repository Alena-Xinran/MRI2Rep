import re
import config
from collections import Counter
from utils.llm import call_llm
from utils.io import load_prompt
import json

def run_step_1_organ(report_text, logger):
    """Step 1: 器官状态初筛"""
    print("  Step 1: Organ Classification...")
    raw_prompt = load_prompt("PROMPT_STEP_1_ORGAN.txt")
    prompt = raw_prompt.replace("{report_text}", report_text)
    
    response = call_llm(prompt, temperature=0.0)
    content = response.choices[0].message.content.strip() if response else ""
    
    # 正则提取
    match = re.search(r"FINAL_RESULT:\s*(.+)", content)
    status = match.group(1).strip() if match else "Unknown"
    
    logger.log_step("step_1_organ", prompt, response, extracted_info=status)
    return status

def run_step_2_verification(report_text, previous_status, logger):
    """Step 2: 器官状态校验"""
    print("  Step 2: Organ Verification...")
    raw_prompt = load_prompt("PROMPT_STEP_2_ORGAN_CHECK.txt")
    prompt = raw_prompt.replace("{report_text}", report_text).replace("{previous_status}", previous_status)
    
    response = call_llm(prompt, temperature=0.0)
    content = response.choices[0].message.content.strip() if response else ""
    
    match = re.search(r"CONFIRMED_RESULT:\s*(.+)", content)
    final_status = match.group(1).strip() if match else previous_status
    
    logger.log_step("step_2_verification", prompt, response, extracted_info=final_status)
    return final_status

def run_step_3_tumor_extraction(report_text, logger):
    """Step 3: 肿瘤/病灶提取 (含多数投票机制)"""
    rounds = config.CONSISTENCY_ROUNDS
    print(f"  Step 3: Tumor Extraction ({rounds} rounds)...")
    
    raw_prompt = load_prompt("PROMPT_STEP_3_TUMOR_EXTRACTION.txt")
    prompt = raw_prompt.replace("{report_text}", report_text)
    
    extraction_results = []
    # 用于投票的标准化字符串列表
    normalized_results_for_voting = []
    
    # 循环执行
    for i in range(1, rounds + 1):
        print(f"    - Run {i}/{rounds}...")
        # 多轮时增加随机性
        temp = 0.2 if rounds > 1 else 0.0
        response = call_llm(prompt, temperature=temp)
        content = response.choices[0].message.content.strip() if response else ""
        
        # 提取记录
        records = re.findall(r"TUMOR_RECORD:\s*(.+)", content)
        record_str = "\n".join(records) if records else "NONE"
        
        logger.log_step(f"step_3_extraction_run_{i}", prompt, response, extracted_info=record_str)
        
        extraction_results.append(records)
        
        # 生成一个可哈希的标准化字符串用于投票
        # 逻辑：排序 -> 去空白 -> 转为 tuple -> 转字符串
        # 例如: "TypeA|Loc1|1" 和 " TypeA | Loc1 | 1 " 应该视为相同
        sorted_records = sorted([r.strip() for r in records])
        voting_key = json.dumps(sorted_records) # 使用 JSON 字符串作为唯一的 Key
        normalized_results_for_voting.append(voting_key)

    # === 多数投票逻辑 (Majority Voting) ===
    
    # 统计每种结果出现的次数
    vote_counts = Counter(normalized_results_for_voting)
    
    # 找到票数最多的结果 (most_common 返回 [(key, count), ...])
    # 如果有平票，most_common 会按遇到顺序返回，所以默认偏向较早出现的
    winner_key, winner_count = vote_counts.most_common(1)[0]
    
    # 找到这个 winner 对应的原始记录列表 (从 extraction_results 里找一个匹配的)
    final_records = []
    for idx, key in enumerate(normalized_results_for_voting):
        if key == winner_key:
            final_records = extraction_results[idx]
            break
    
    # === 一致性检查 ===
    # 如果 winner 的票数等于总轮数，说明完全一致
    is_consistent = (winner_count == rounds)
    
    # 记录统计信息
    consistency_stats = {
        "total_rounds": rounds,
        "winner_count": winner_count,
        "details": dict(vote_counts)
    }

    print(f"    - Voting Result: Winner appeared {winner_count}/{rounds} times.")
    print(f"    - Consistency: {'PASS' if is_consistent else 'FAIL'}")
    
    return final_records, is_consistent, consistency_stats

def parse_tumor_records(raw_records):
    """将文本记录转为结构化 Dict"""
    structured = []
    for record in raw_records:
        if "NONE" in record: continue
        parts = [p.strip() for p in record.split('|')]
        if len(parts) == 3:
            structured.append({
                "type": parts[0],
                "position": parts[1],
                "quantity": parts[2]
            })
    return structured
