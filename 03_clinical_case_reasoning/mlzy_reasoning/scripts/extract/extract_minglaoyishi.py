"""
从名老中医CSV数据中提取辨证病案，转为与 train_grpo_600.json 相同的 raw_data 格式。
输出：名老中医_extracted.json
"""

import csv
import json
import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_DIR = os.path.join(PROJECT_ROOT, "名老中医", "Csv")
ALL_DATA_CSV = os.path.join(CSV_DIR, "全部数据.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "名老中医_extracted.json")

# ==========================================
# 1. 数据清洗工具函数
# ==========================================

def clean_text(text: str) -> str:
    """去除多余空格、换行，标准化标点"""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def dedup_semicolon_list(text: str) -> list[str]:
    """将分号分隔的字段去重，返回唯一值列表"""
    if not text:
        return []
    parts = [p.strip() for p in text.split(';') if p.strip()]
    # 保持顺序去重
    seen = set()
    result = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def extract_age(age_str: str) -> str:
    """提取年龄，格式化为 'XX岁'"""
    if not age_str:
        return ""
    age_str = age_str.strip()
    # 已经是纯数字
    if age_str.isdigit():
        return f"{age_str}岁"
    return age_str


def dedup_tongue_pulse(text: str) -> str:
    """去除多次就诊导致的重复舌脉描述"""
    if not text:
        return ""
    # 按逗号或分号拆分，去重
    segments = re.split(r'[,，;；]', text)
    seen = set()
    result = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg not in seen:
            seen.add(seg)
            result.append(seg)
    return '，'.join(result)


def build_望闻切诊(row: dict) -> str:
    """
    从舌脉象、舌质舌态、脉等字段构造 '中医望闻切诊' 文本。
    模仿 train_grpo_600.json 的格式。
    """
    # 舌象
    she = clean_text(row.get('舌质舌态', ''))
    if not she:
        she_mai = clean_text(row.get('舌脉象', ''))
        if she_mai:
            she_parts = []
            for seg in re.split(r'[;；]', she_mai):
                seg = seg.strip()
                if not seg:
                    continue
                if any(kw in seg for kw in ['舌', '苔', '齿痕']):
                    she_parts.append(seg)
                elif not any(kw in seg for kw in ['脉', '六脉']):
                    she_parts.append(seg)
            she = '，'.join(she_parts) if she_parts else ''
    # 去重多次就诊的重复描述
    she = dedup_tongue_pulse(she)
    # 清除舌脉象中未拆分的长字符串（如 '舌暗舌边齿痕苔厚腻微黄六脉弦数...' 没被正确拆分）
    if she and len(she) > 30:
        # 尝试重新拆分连续的舌象描述
        import re as _re
        she = _re.sub(r'(舌质|舌伴质|舌淡|舌暗|舌红|舌胖)', r'，\1', she).lstrip('，')
        she = dedup_tongue_pulse(she)

    # 脉象
    mai = clean_text(row.get('脉', ''))
    if not mai:
        she_mai = clean_text(row.get('舌脉象', ''))
        if she_mai:
            for seg in re.split(r'[;；]', she_mai):
                seg = seg.strip()
                if any(kw in seg for kw in ['脉', '六脉']):
                    mai = seg
                    break
    mai = dedup_tongue_pulse(mai)

    # 体格检查
    tg = clean_text(row.get('体格检查', ''))

    # 构造望闻切诊文本
    wangwen = []
    if she:
        wangwen.append(f"舌象：{she}")
    if mai:
        if not mai.startswith('脉'):
            mai = f"脉{mai}"
        wangwen.append(f"脉象：{mai}")
    if tg and tg != '苔舌':  # 过滤无效体格检查
        wangwen.append(f"体格检查：{tg}")

    return '。'.join(wangwen) + '。' if wangwen else ""


def clean_symptoms(symptom_str: str) -> list[str]:
    """清洗症状列表：去重、去除伪造拼接（如'口易怒'实为拼接错误）"""
    if not symptom_str:
        return []
    symptoms = dedup_semicolon_list(symptom_str)
    cleaned = []
    for s in symptoms:
        s = s.strip()
        if not s or len(s) > 20:
            continue
        # 过滤明显的拼接错误：以'口'开头但不是正常中医术语
        if s.startswith('口') and len(s) > 1:
            valid_mouth = ['口干', '口渴', '口苦', '口臭', '口淡', '口黏', '口甘', '口酾',
                          '口干口苦', '口干口渴', '口干燥症', '口口苦',
                          '口唇', '口腔', '口舌']
            if not any(s.startswith(v) for v in valid_mouth):
                continue  # 跳过拼接错误如 '口易怒'、'口焦虑'
        cleaned.append(s)
    return cleaned


def build_symptoms(row: dict) -> str:
    """
    构造完整症状描述，合并 症状 + 二便情况 + 睡眠 等。
    """
    parts = []

    # 主要症状
    zhengzhuang = clean_text(row.get('症状', ''))
    if zhengzhuang:
        symptoms = clean_symptoms(zhengzhuang)
        if symptoms:
            parts.append('，'.join(symptoms))

    # 睡眠
    sleep = clean_text(row.get('睡眠', ''))
    if sleep:
        parts.append(sleep)

    # 二便
    erbian = clean_text(row.get('二便情况', ''))
    if erbian:
        # 去重二便描述中的分号
        erbian_parts = dedup_semicolon_list(erbian)
        parts.append('，'.join(erbian_parts))
    else:
        dabian = clean_text(row.get('大便', ''))
        xiaobian = clean_text(row.get('小便', ''))
        if dabian:
            parts.append(f"大便：{dabian}")
        if xiaobian:
            parts.append(f"小便：{xiaobian}")

    return '，'.join(parts) if parts else ""


def build_prescription(row: dict) -> str:
    """从处方明细提取药物列表，格式化为 ['药1', '药2', ...] 字符串"""
    chufang = clean_text(row.get('处方明细', ''))
    if not chufang:
        return ""
    herbs = dedup_semicolon_list(chufang)
    return str(herbs)


def build_syndrome(row: dict) -> str:
    """从中医症候名称提取证型，用 | 连接"""
    zhenghou = clean_text(row.get('中医症候名称', ''))
    if not zhenghou:
        return ""
    types = dedup_semicolon_list(zhenghou)
    return '|'.join(types)


def build_disease(row: dict) -> str:
    """提取中医疾病诊断名称"""
    zhenduan = clean_text(row.get('中医诊断名称', ''))
    if not zhenduan:
        # 回退到中医诊断
        zhenduan = clean_text(row.get('中医诊断', ''))
    if not zhenduan:
        return ""
    diseases = dedup_semicolon_list(zhenduan)
    return '|'.join(diseases)


# ==========================================
# 2. 质量评分
# ==========================================

def quality_score(row: dict) -> int:
    """对一条数据评分，分数越高数据越完整"""
    score = 0
    if row.get('主诉', '').strip():
        score += 3
    if row.get('症状', '').strip():
        score += 3
    if row.get('中医症候名称', '').strip():
        score += 3
    if row.get('处方明细', '').strip():
        score += 2
    if row.get('舌质舌态', '').strip() or row.get('舌脉象', '').strip():
        score += 2
    if row.get('脉', '').strip():
        score += 1
    if row.get('体格检查', '').strip():
        score += 1
    if row.get('睡眠', '').strip():
        score += 1
    if row.get('二便情况', '').strip() or row.get('大便', '').strip():
        score += 1
    return score


# ==========================================
# 3. 主提取逻辑
# ==========================================

def extract_cases():
    """从全部数据CSV提取病案"""
    print(f"读取: {ALL_DATA_CSV}")

    all_rows = []
    with open(ALL_DATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    print(f"CSV总行数: {len(all_rows)}")

    # 筛选：至少有 症状 + 证候 + 处方 + 主诉
    qualified = []
    for row in all_rows:
        has_symptom = bool(row.get('症状', '').strip())
        has_zhenghou = bool(row.get('中医症候名称', '').strip())
        has_chufang = bool(row.get('处方明细', '').strip())
        has_zhusu = bool(row.get('主诉', '').strip())
        if has_symptom and has_zhenghou and has_chufang and has_zhusu:
            qualified.append(row)

    print(f"四项齐全（症状+证候+处方+主诉）: {len(qualified)}")

    # 按患者编号去重，每个患者只保留质量最高的一条
    patient_best = {}
    for row in qualified:
        pid = row.get('患者编号', '')
        score = quality_score(row)
        if pid not in patient_best or score > patient_best[pid][1]:
            patient_best[pid] = (row, score)

    best_rows = [v[0] for v in patient_best.values()]
    print(f"按患者去重后: {len(best_rows)}")

    # 转换为 train_grpo_600.json 的 raw_data 格式
    results = []
    for idx, row in enumerate(best_rows):
        pid = row.get('患者编号', str(idx))
        gender = row.get('性别', '').replace('性', '')  # "男性" -> "男"
        age = extract_age(row.get('就诊年龄', ''))
        zhusu = clean_text(row.get('主诉', ''))
        symptoms = build_symptoms(row)
        wangwen = build_望闻切诊(row)
        disease = build_disease(row)
        syndrome = build_syndrome(row)
        prescription = build_prescription(row)
        tg = clean_text(row.get('体格检查', ''))

        raw_data = {
            "ID": f"MLZ_{pid}",
            "性别": gender,
            "年龄": age,
            "主诉": zhusu,
            "症状": symptoms,
            "中医望闻切诊": wangwen,
            "体格检查": tg if tg else "",
            "疾病": disease,
            "证型": syndrome,
            "处方": prescription,
        }

        results.append(raw_data)

    return results


def main():
    cases = extract_cases()

    # 按质量排序：优先单次就诊、信息完整的病案
    def case_quality(case):
        syndrome_count = len(case['证型'].split('|'))
        disease_count = len(case['疾病'].split('|')) if case['疾病'] else 0
        # 证型过多（>4）说明是多次就诊合并，质量低
        penalty = max(0, syndrome_count - 4) * 10 + max(0, disease_count - 3) * 5
        # 奖励：有舌脉、有主诉质量、症状丰富
        bonus = 0
        if case['中医望闻切诊']:
            bonus += 5
        if len(case['症状']) > 20:
            bonus += 3
        if case['体格检查']:
            bonus += 2
        return bonus - penalty

    cases.sort(key=case_quality, reverse=True)

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"\n提取完成！共 {len(cases)} 条病案 -> {OUTPUT_FILE}")

    # 打印统计
    print("\n=== 数据质量统计 ===")
    has_wangwen = sum(1 for c in cases if c['中医望闻切诊'])
    has_tg = sum(1 for c in cases if c['体格检查'])
    has_disease = sum(1 for c in cases if c['疾病'])
    multi_syn = sum(1 for c in cases if '|' in c['证型'])
    print(f"有望闻切诊: {has_wangwen}/{len(cases)}")
    print(f"有体格检查: {has_tg}/{len(cases)}")
    print(f"有疾病诊断: {has_disease}/{len(cases)}")
    print(f"多证型: {multi_syn}/{len(cases)}")

    # 打印前3条示例
    print("\n=== 前3条示例 ===")
    for i, case in enumerate(cases[:3]):
        print(f"\n--- 第{i+1}条 ---")
        for k, v in case.items():
            print(f"  {k}: {str(v)[:100]}")


if __name__ == "__main__":
    main()
