"""
TRL GRPOTrainer 自定义奖励函数 — 中医辨证结构化训练

TRL 奖励函数签名：reward_fn(completions, **kwargs) -> list[float]
- completions: list[str]，模型生成的文本
- kwargs 中包含 dataset 的其他列，如 ground_truth

模型输出格式：<think>四诊提取 + 辨证参数推导 + 证型判定</think>["证型1", "证型2"]
奖励 = 0.20 * format + 0.40 * syndrome + 0.40 * bianzheng
"""
from __future__ import annotations
import json
import os
import re


# ==========================================
# 常量（与 reward_functions.py 一致）
# ==========================================
BAGANG_FIELDS = ["biao_li", "han_re", "xu_shi", "yin_yang"]
BAGANG_OPTIONS = {
    "biao_li": ["表", "里", "半表半里", "表里同病", "未定"],
    "han_re": ["寒", "热", "寒热错杂", "未定"],
    "xu_shi": ["虚", "实", "虚实夹杂", "未定"],
    "yin_yang": ["阴", "阳", "阴阳并见", "未定"],
}
BAGANG_CN = {"表里": "biao_li", "寒热": "han_re", "虚实": "xu_shi", "阴阳": "yin_yang"}

ZANGFU_CN = {"心": "xin", "肝": "gan", "脾": "pi", "肺": "fei", "肾": "shen"}
ZANGFU_FIELDS = list(ZANGFU_CN.values())

QI_BLOOD_CN = {
    "气虚": "qi_xu", "气陷": "qi_xian", "气脱": "qi_tuo",
    "气滞": "qi_zhi", "气逆": "qi_ni", "气闭": "qi_bi",
    "血虚": "xue_xu", "血瘀": "xue_yu", "血热": "xue_re", "血寒": "xue_han",
    "津液亏虚": "jin_ye_kui_xu", "痰证": "tan_zheng", "饮证": "yin_zheng",
    "水停证": "shui_ting_zheng", "内湿证": "nei_shi_zheng",
}
QI_BLOOD_FIELDS = list(QI_BLOOD_CN.values())

PATHO_CN = {"风": "feng", "寒": "han", "暑": "shu", "湿": "shi", "燥": "zao", "火": "huo"}
PATHO_FIELDS = list(PATHO_CN.values())

SIZHEN_KEYWORDS = ["望诊", "闻诊", "问诊", "切诊"]
BAGANG_LABELS = ["表里", "寒热", "虚实", "阴阳"]
ZANGFU_LABELS = ["心", "肝", "脾", "肺", "肾"]
QI_BLOOD_LABELS = list(QI_BLOOD_CN.keys())
PATHO_LABELS = ["风", "寒", "暑", "湿", "燥", "火"]

FORMAT_GROUPS = [
    ("sizhen", SIZHEN_KEYWORDS, "keyword"),
    ("bagang", BAGANG_LABELS, "colon"),
    ("zangfu", ZANGFU_LABELS, "equal"),
    ("qi_blood", QI_BLOOD_LABELS, "equal"),
    ("patho", PATHO_LABELS, "equal"),
]


# ==========================================
# 国标证型树距离（GB/T 15657-2021）
# ==========================================
_BTREE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zhenghou_btree.json")
with open(_BTREE_PATH, "r", encoding="utf-8") as _f:
    _BTREE_DATA = json.load(_f)
NAME_TO_CODE: dict[str, str] = _BTREE_DATA["name_to_code"]

PATHO_KEYWORDS = [
    "气虚", "气陷", "气脱", "气滞", "气逆", "气闭",
    "血虚", "血瘀", "血热", "血寒",
    "阴虚", "阳虚", "阳亢",
    "痰", "湿", "水停", "饮",
    "风", "寒", "暑", "燥", "火", "热",
]


def _code_depth(code: str) -> int:
    parts = code.lstrip("B").strip(".").split(".")
    return len([p for p in parts if p])


def _lca_depth(code_a: str, code_b: str) -> int:
    parts_a = code_a.lstrip("B").strip(".").split(".")
    parts_b = code_b.lstrip("B").strip(".").split(".")
    shared = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            shared += 1
        else:
            break
    return shared


def _tree_similarity(code_a: str, code_b: str) -> float:
    if code_a == code_b:
        return 1.0
    lca_d = _lca_depth(code_a, code_b)
    da = _code_depth(code_a)
    db = _code_depth(code_b)
    if da + db == 0:
        return 0.0
    return 2 * lca_d / (da + db)


def _extract_patho_features(name: str) -> set[str]:
    return {kw for kw in PATHO_KEYWORDS if kw in name}


def _name_feature_similarity(name_a: str, name_b: str) -> float:
    fa = _extract_patho_features(name_a)
    fb = _extract_patho_features(name_b)
    if not fa and not fb:
        return 0.0
    intersection = fa & fb
    union = fa | fb
    if not union:
        return 0.0
    return min(len(intersection) / len(union) * 0.8, 0.8)


def syndrome_similarity(name_a: str, name_b: str) -> float:
    if name_a == name_b:
        return 1.0

    code_a = NAME_TO_CODE.get(name_a)
    code_b = NAME_TO_CODE.get(name_b)

    if not code_a or not code_b:
        return 0.0

    sim = _tree_similarity(code_a, code_b)

    if sim > 0.05:
        return sim

    cat_a = code_a[:3]
    cat_b = code_b[:3]
    if {cat_a, cat_b} == {"B03", "B04"}:
        return _name_feature_similarity(name_a, name_b)

    return sim


# ==========================================
# 解析工具
# ==========================================
def _extract_think(text: str):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1)
    # TRL 只传 generation 部分，<think> 在 prompt 末尾，completion 里只有 </think>
    if "</think>" in text:
        return text.split("</think>", 1)[0]
    return ""


def _extract_syndrome(text: str):
    try:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            # 展平嵌套 list 并只保留字符串元素
            flat = []
            for item in parsed:
                if isinstance(item, str):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(s for s in item if isinstance(s, str))
            return flat if flat else None
        if isinstance(parsed, dict):
            val = parsed.get("syndrome", parsed.get("syndromes"))
            if isinstance(val, list):
                return [s for s in val if isinstance(s, str)] or None
            return None
        return None
    except Exception:
        return None


def _parse_bagang_from_think(think: str) -> dict:
    result = {}
    for cn_name, en_key in BAGANG_CN.items():
        m = re.search(rf"{cn_name}\s*[：:]\s*([^。\n]+)", think)
        if m:
            val = m.group(1).strip()
            for opt in sorted(BAGANG_OPTIONS[en_key], key=len, reverse=True):
                if val.startswith(opt):
                    result[en_key] = opt
                    break
    return result


def _parse_numeric_from_think(think: str, cn_map: dict) -> dict:
    result = {}
    for cn_name, en_key in cn_map.items():
        m = re.search(rf"{re.escape(cn_name)}\s*[=＝:：]\s*([0-3])", think)
        if m:
            result[en_key] = int(m.group(1))
    return result


# ==========================================
# 子奖励
# ==========================================
def _repetition_penalty(text: str) -> float:
    """检测重复退化，返回 [0, 1] 惩罚系数，0=无重复，1=完全重复"""
    if len(text) < 50:
        return 0.0
    worst = 0.0
    # 短 pattern 连续重复（如 【】【】【】...）
    compressed = re.sub(r'\s+', '', text)
    for m in re.finditer(r'(.{2,20}?)\1{4,}', compressed):
        ratio = len(m.group(0)) / len(compressed)
        worst = max(worst, ratio)
    # 连续重复行
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 5:
        max_consec, cur = 1, 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i - 1]:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 1
        if max_consec >= 3:
            worst = max(worst, min(max_consec / len(lines), 0.8))
    return min(worst, 1.0)


# 段落 header 定义：(段落名, header 正则, 段落内关键词, 检测模式, 段落权重)
SECTION_DEFS = [
    ("sizhen",    r"一[、.]\s*四诊提取",    SIZHEN_KEYWORDS,  "keyword", 0.10),
    ("bagang",    r"【八纲】",              BAGANG_LABELS,    "colon",   0.10),
    ("zangfu",    r"【脏腑】",              ZANGFU_LABELS,    "equal",   0.10),
    ("qi_blood",  r"【气血津液】",          QI_BLOOD_LABELS,  "equal",   0.10),
    ("patho",     r"【六淫】",              PATHO_LABELS,     "equal",   0.10),
    ("judgment",  r"三[、.]\s*证型判定",     [],               "none",    0.05),
]


def _split_sections(think: str) -> dict[str, str]:
    """按 header 切分 think 内容，返回 {段落名: 段落文本}"""
    boundaries = []
    for name, pattern, _kw, _mode, _w in SECTION_DEFS:
        m = re.search(pattern, think)
        if m:
            boundaries.append((m.start(), name))
    boundaries.sort(key=lambda x: x[0])

    sections = {}
    for i, (start, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(think)
        sections[name] = think[start:end]
    return sections


def _format_score(output: str) -> float:
    score = 0.0
    has_open = "<think>" in output
    has_close = "</think>" in output
    # TRL completion 可能不含 <think>（在 prompt 末尾），只要有 </think> 就算格式正确
    if has_close:
        score += 0.05
    elif has_open:
        score += 0.025
    else:
        return 0.0

    think = _extract_think(output)
    if not think:
        return score

    sections = _split_sections(think)

    # --- 段落结构分（0.30）：header 存在 + 顺序正确 ---
    # header 存在：每个段落有 header 得分
    header_score = len(sections) / len(SECTION_DEFS) * 0.15
    # 顺序检查：找到的段落是否按定义顺序排列
    expected_order = [name for name, *_ in SECTION_DEFS]
    found_order = [name for name in expected_order if name in sections]
    found_in_think = []
    for name, pattern, *_ in SECTION_DEFS:
        m = re.search(pattern, think)
        if m:
            found_in_think.append((m.start(), name))
    found_in_think.sort(key=lambda x: x[0])
    actual_order = [name for _, name in found_in_think]
    if len(actual_order) >= 2 and actual_order == found_order:
        order_score = 0.15
    elif len(actual_order) >= 2:
        # 部分有序：计算最长有序子序列比例
        correct = sum(
            1 for i in range(len(actual_order) - 1)
            if expected_order.index(actual_order[i]) < expected_order.index(actual_order[i + 1])
        )
        order_score = correct / (len(actual_order) - 1) * 0.15
    else:
        order_score = 0.0
    score += header_score + order_score

    # --- 段落内关键词分（0.30）：只在对应段落内检测 ---
    for name, _pattern, keywords, mode, weight in SECTION_DEFS:
        if not keywords or name not in sections:
            continue
        section_text = sections[name]
        found = 0
        for k in keywords:
            if mode == "keyword":
                m = re.search(re.escape(k), section_text)
            elif mode == "colon":
                m = re.search(rf"{re.escape(k)}\s*[：:]", section_text)
            else:
                m = re.search(rf"{re.escape(k)}\s*[=＝]\s*\d", section_text)
            if m:
                found += 1
        score += (found / len(keywords)) * weight

    # --- 三、证型判定段 bonus（0.05 已含在 header_score 中） ---

    # --- 证型 JSON 列表（0.15）---
    syn = _extract_syndrome(output)
    if isinstance(syn, list) and len(syn) > 0:
        score += 0.15

    # --- 重复惩罚（全局乘数） ---
    rep = _repetition_penalty(think)
    score *= (1.0 - rep)

    return score


def _syndrome_score(output: str, gt: dict) -> float:
    pred = _extract_syndrome(output)
    if not isinstance(pred, list) or not pred:
        return 0.0
    gt_list = gt.get("syndrome", gt.get("syndromes", []))
    if not gt_list:
        return 0.0

    if set(pred) == set(gt_list):
        return 1.0

    soft_p = sum(max(syndrome_similarity(p, g) for g in gt_list) for p in pred) / len(pred)
    soft_r = sum(max(syndrome_similarity(g, p) for p in pred) for g in gt_list) / len(gt_list)

    if soft_p + soft_r == 0:
        return 0.0
    return 2 * soft_p * soft_r / (soft_p + soft_r)


def _bianzheng_score(output: str, gt: dict) -> float:
    think = _extract_think(output)
    if not think:
        return 0.0

    gt_bz = gt.get("bianzheng", {})

    gt_bg = gt_bz.get("bagang", {})
    pred_bg = _parse_bagang_from_think(think)
    bg_match = sum(1 for f in BAGANG_FIELDS if pred_bg.get(f) == gt_bg.get(f))
    bg_score = bg_match / len(BAGANG_FIELDS)
    bg_coverage = len(pred_bg) / len(BAGANG_FIELDS)

    pred_zf = _parse_numeric_from_think(think, ZANGFU_CN)
    pred_qb = _parse_numeric_from_think(think, QI_BLOOD_CN)
    pred_pf = _parse_numeric_from_think(think, PATHO_CN)

    weighted_ae, weight_sum = 0.0, 0.0
    nonzero_hit, nonzero_weight = 0.0, 0.0
    parsed_numeric, total_numeric = 0, 0
    for grp, fields, pred_sub in [
        ("zangfu", ZANGFU_FIELDS, pred_zf),
        ("qi_blood_fluid", QI_BLOOD_FIELDS, pred_qb),
        ("patho_factors", PATHO_FIELDS, pred_pf),
    ]:
        gt_sub = gt_bz.get(grp, {})
        for f in fields:
            total_numeric += 1
            gv = gt_sub.get(f, 0)
            has_pred = f in pred_sub
            if has_pred:
                parsed_numeric += 1
            pv = pred_sub.get(f, 0)
            if not isinstance(gv, (int, float)):
                gv = 0
            if not isinstance(pv, (int, float)):
                pv = 0

            # Standard non-zero parameters are rarer and more important than zeros.
            # Weight by standard intensity to reduce dilution from many zero labels.
            weight = 1.0 + max(float(gv), 0.0)
            weighted_ae += weight * abs(gv - pv)
            weight_sum += weight

            if gv > 0:
                nonzero_weight += float(gv)
                if pv > 0:
                    nonzero_hit += float(gv)

    wmae = weighted_ae / max(weight_sum, 1.0)
    wmae_score = max(0.0, 1.0 - wmae / 1.5)
    nonzero_score = nonzero_hit / nonzero_weight if nonzero_weight > 0 else 1.0
    num_score = 0.70 * wmae_score + 0.30 * nonzero_score
    num_coverage = parsed_numeric / max(total_numeric, 1)
    coverage = 0.30 * bg_coverage + 0.70 * num_coverage

    return (0.30 * bg_score + 0.70 * num_score) * coverage


def _compute_single(completion: str, ground_truth_str: str) -> float:
    """计算单条样本的奖励分"""
    try:
        gt = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
    except Exception:
        gt = {}

    fmt = _format_score(completion)
    syn = _syndrome_score(completion, gt)
    bz = _bianzheng_score(completion, gt)

    return 0.20 * fmt + 0.40 * syn + 0.40 * bz


# ==========================================
# TRL 接口：reward_fn
# ==========================================
def reward_fn(completions, ground_truth: list[str] | None = None, **kwargs) -> list[float]:
    """
    TRL GRPOTrainer 调用的奖励函数入口

    Args:
        completions: 模型生成结果，可能是 list[str] 或 list[list[dict]]（chat 格式）
        ground_truth: ground truth JSON 字符串列表（来自 dataset 的 ground_truth 列）
    Returns:
        list[float]: 每条样本的奖励分数 [0, 1]
    """
    # TRL 新版传 chat message 格式，提取文本
    texts = []
    for c in completions:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list) and len(c) > 0 and isinstance(c[0], dict):
            texts.append(c[-1].get("content", ""))
        else:
            texts.append(str(c))

    if ground_truth is None:
        ground_truth = ["{}"] * len(texts)

    return [_compute_single(t, gt) for t, gt in zip(texts, ground_truth)]


# 同时兼容 verl 接口，方便调试
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _compute_single(solution_str, ground_truth)
