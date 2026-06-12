"""
mlzy 数据四诊补足 + 辨证参数复核脚本

输入：bianzheng_mlzy_train.jsonl (1402条) + bianzheng_mlzy_test.jsonl (295条)
输出：*_verified.jsonl（不覆盖原文件）

两步合一：单次 API 调用同时补足缺失四诊 + 复核辨证参数

用法：python3 verify_mlzy.py
"""
import json
import os
import re
import time
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", "?")
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable, 1):
            if i % 10 == 0 or i == total:
                print(f"\r  {desc} {i}/{total}", end="", flush=True)
            yield item
        print()

# ==========================================
# 配置
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
BASE_URL = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL") or "https://cc.580ai.net"
MODEL = os.getenv("ANTHROPIC_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME") or "claude-opus-4-6"
CONCURRENCY = 10
MAX_TOKENS = 2560

PROGRESS_PATH = os.path.join(PROJECT_ROOT, "data", "progress", "verify_mlzy_progress.json")

INPUT_FILES = [
    ("bianzheng_mlzy_train.jsonl", "bianzheng_mlzy_train_verified.jsonl"),
    ("bianzheng_mlzy_test.jsonl", "bianzheng_mlzy_test_verified.jsonl"),
]

write_lock = threading.Lock()

# ==========================================
# 从 verify_bianzheng.py 复用 mapping 规则
# ==========================================
def load_mapping_rules():
    with open(os.path.join(CONFIG_DIR, "mapping_table.json"), "r", encoding="utf-8") as f:
        mt = json.load(f)

    lines = []

    lines.append("## 八纲辨证选项")
    bg = mt["bagang_mapping"]
    for dim in ["biao_li", "han_re", "xu_shi", "yin_yang"]:
        opts = mt["_说明"]["八纲选项"][dim]
        lines.append(f"- {dim}: {opts}")
        for opt_name, opt_data in bg[dim].items():
            keywords = opt_data["关键证据"][:12]
            rule = opt_data.get("判定规则", "")
            lines.append(f"  {opt_name}: {','.join(keywords)}... 规则:{rule}")

    lines.append("\n## 脏腑辨证 (0-3分)")
    lines.append("只评估: xin, gan, pi, fei, shen")
    zf = mt["zangfu_mapping"]
    for organ in ["xin", "gan", "pi", "fei", "shen"]:
        data = zf[organ]["症状体征"]
        s3 = data.get("强证据_3", [])[:6]
        s2 = data.get("中等证据_2", [])[:8]
        s1 = data.get("弱证据_1", [])[:4]
        lines.append(f"- {organ}: 强({','.join(s3)}) 中({','.join(s2)}) 弱({','.join(s1)})")

    lines.append("\n## 气血津液辨证 (0-3分)")
    qb = mt["qi_blood_fluid_mapping"]
    for key in ["qi_xu","qi_xian","qi_tuo","qi_zhi","qi_ni","qi_bi",
                "xue_xu","xue_yu","xue_re","xue_han",
                "jin_ye_kui_xu","tan_zheng","yin_zheng","shui_ting_zheng","nei_shi_zheng"]:
        data = qb[key]["关键证据"]
        name = qb[key]["名称"]
        s3 = data.get("强证据_3", [])[:5]
        s2 = data.get("中等证据_2", [])[:6]
        lines.append(f"- {key}({name}): 强({','.join(s3)}) 中({','.join(s2)})")

    lines.append("\n## 病理因素 (0-3分)")
    pf = mt["patho_factors_mapping"]
    for key in ["feng","han","shu","shi","zao","huo"]:
        data = pf[key]["关键证据"]
        name = pf[key]["名称"]
        s3 = data.get("强证据_3", [])[:5]
        s2 = data.get("中等证据_2", [])[:6]
        lines.append(f"- {key}({name}): 强({','.join(s3)}) 中({','.join(s2)})")

    return "\n".join(lines)


MAPPING_RULES = load_mapping_rules()

SYSTEM_PROMPT = f"""你是中医四诊补全与辨证参数审核专家。你有两个任务：

## 任务一：补全中医望闻切诊并拆分四诊
原始数据中的"中医望闻切诊"字段往往不完整（例如只有舌象和脉象，缺少面色、神态、形体、语声、气息等）。
请根据原始数据（主诉、症状、中医望闻切诊原文）：
1. 先补全"中医望闻切诊"为一句完整描述，格式示例：
   "中医望闻切诊：表情自然，面色红润，形体正常,语气清,气息平；无异常气味,舌质红、苔黄厚腻,舌下脉络迂曲,舌边有瘀斑，脉弦滑数"
   要求包含：面色、神态、形体、语声、气息、气味、舌象、脉象，用逗号/分号连接为一句话
2. 然后将补全后的内容拆分到四诊：
   - 望诊(wang)：面色、神态、形体、舌象等目视所见
   - 闻诊(wen)：语声、气息、口气等听觉嗅觉所得
   - 问诊(wen_zhen)：主诉、症状、病史等问询所得
   - 切诊(qie)：脉象、按诊等触诊所得

补全原则：
- 原始数据中已有的信息必须保留，不能丢弃
- 原始数据中未提及的内容，根据症状和证型合理推断；如确实无法推断，填"未见明显异常"
- 不能凭空编造与原始数据矛盾的症状

## 任务二：复核辨证参数
基于补全后的完整四诊，检查辨证参数是否合理，发现矛盾则改正。

评分规则参考：
{MAPPING_RULES}

评分原则：
- 数值类参数(0-3): 0=无证据, 1=弱证据/轻度相关, 2=中等证据/明显相关, 3=强证据/主导因素
- 【最重要】所有赋值必须且只能基于四诊中的实际症状/体征，四诊中未提及的症状绝不能作为赋值依据
- 证型仅供参考背景，不能作为赋值依据。例如：不能因为证型含"血瘀"就判定虚实夹杂，必须在四诊中找到实邪证据（如疼痛拒按、舌苔厚腻、脉有力等）才能判实或虚实夹杂
- 反过来，即使证型未涉及某脏腑或病理因素，只要四诊中有该维度的明确证据，也必须如实赋分
- 八纲参数从给定选项中选择，判定依据同样必须来自四诊
- 如果原参数合理则保持不变，只改正有矛盾的部分

输出要求：
- 只输出一个JSON对象，包含四个字段: "zhongyi_wangwenqiezhen", "four_diagnosis", "bianzheng", "explanations"
- zhongyi_wangwenqiezhen: 补全后的完整中医望闻切诊描述（一段文字）
- four_diagnosis: 从补全后的望闻切诊拆分出的 wang/wen/wen_zhen/qie
- bianzheng 结构与输入完全一致
- explanations 对每个辨证参数给出简短理由(一句话)
- 不要输出任何其他文字，只输出JSON"""


# ==========================================
# 调用 Claude API
# ==========================================
def call_api(user_content: str, retries: int = 3) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }).encode("utf-8")
    url = f"{BASE_URL.rstrip('/')}/v1/messages"

    for attempt in range(retries):
        try:
            req = Request(url, data=body, method="POST", headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            })
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt * 5)
            else:
                time.sleep(2)
        except (URLError, Exception):
            time.sleep(2)
    return None


# ==========================================
# 检测四诊缺失
# ==========================================
def is_missing(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip()
    return t.startswith("未") or t.startswith("无法") or t == "无"


# ==========================================
# 修复异常 converted 结构 (record 747, 679)
# ==========================================
def fix_converted(record: dict) -> dict:
    c = record["converted"]
    if "four_diagnosis" in c:
        return record

    for key, val in c.items():
        if isinstance(val, dict) and "four_diagnosis" in val:
            record["converted"] = val
            record["converted"]["case_id"] = record.get("case_id", key)
            return record

    return record


# ==========================================
# 构造 user prompt
# ==========================================
def build_user_prompt(record: dict) -> str:
    c = record["converted"]
    fd = c["four_diagnosis"]
    bz = c["bianzheng"]
    syns = c["syndrome"]
    raw = record["raw_data"]

    parts = []

    parts.append("## 原始数据")
    parts.append(f"主诉: {raw.get('主诉', '无')}")
    parts.append(f"症状: {raw.get('症状', '无')}")
    parts.append(f"中医望闻切诊(原文): {raw.get('中医望闻切诊', '无')}")

    parts.append(f"\n## 证型: {', '.join(syns)}")

    parts.append(f"\n## 当前辨证参数")
    parts.append(json.dumps(bz, ensure_ascii=False, indent=2))

    parts.append("\n请：1) 根据原始数据补全中医望闻切诊并拆分为望闻问切四诊；2) 基于补全后的四诊复核辨证参数。输出完整JSON（含zhongyi_wangwenqiezhen、four_diagnosis、bianzheng、explanations）。")
    return "\n".join(parts)


# ==========================================
# 解析 API 返回
# ==========================================
def parse_response(text: str) -> dict | None:
    text = text.strip()

    m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
    if end == -1:
        return None

    try:
        obj = json.loads(text[start:end + 1])
        if "bianzheng" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


# ==========================================
# 验证 bianzheng 结构
# ==========================================
BAGANG_OPTIONS = {
    "biao_li": ["表", "里", "半表半里", "表里同病", "未定"],
    "han_re": ["寒", "热", "寒热错杂", "未定"],
    "xu_shi": ["虚", "实", "虚实夹杂", "未定"],
    "yin_yang": ["阴", "阳", "阴阳并见", "未定"],
}


def validate_bianzheng(bz: dict, original: dict) -> dict:
    result = {}

    bg = bz.get("bagang", {})
    result["bagang"] = {}
    for key, opts in BAGANG_OPTIONS.items():
        val = bg.get(key, original["bagang"][key])
        result["bagang"][key] = val if val in opts else original["bagang"][key]

    result["zangfu"] = {}
    for key in ["xin", "gan", "pi", "fei", "shen"]:
        val = bz.get("zangfu", {}).get(key, original["zangfu"][key])
        result["zangfu"][key] = max(0, min(3, int(val))) if isinstance(val, (int, float)) else original["zangfu"][key]

    result["qi_blood_fluid"] = {}
    for key in ["qi_xu","qi_xian","qi_tuo","qi_zhi","qi_ni","qi_bi",
                "xue_xu","xue_yu","xue_re","xue_han",
                "jin_ye_kui_xu","tan_zheng","yin_zheng","shui_ting_zheng","nei_shi_zheng"]:
        val = bz.get("qi_blood_fluid", {}).get(key, original["qi_blood_fluid"][key])
        result["qi_blood_fluid"][key] = max(0, min(3, int(val))) if isinstance(val, (int, float)) else original["qi_blood_fluid"][key]

    result["patho_factors"] = {}
    for key in ["feng", "han", "shu", "shi", "zao", "huo"]:
        val = bz.get("patho_factors", {}).get(key, original["patho_factors"][key])
        result["patho_factors"][key] = max(0, min(3, int(val))) if isinstance(val, (int, float)) else original["patho_factors"][key]

    return result


# ==========================================
# 验证 four_diagnosis
# ==========================================
def validate_four_diagnosis(fd_new: dict, fd_original: dict) -> dict:
    result = {}
    for key in ["wang", "wen", "wen_zhen", "qie"]:
        val = fd_new.get(key, "")
        if val and isinstance(val, str) and val.strip():
            result[key] = val.strip()
        else:
            result[key] = fd_original[key]
    return result


# ==========================================
# 检查证型与辨证参数冲突
# ==========================================
SYNDROME_PARAM_RULES = [
    (lambda s: '气虚' in s, 'qi_blood_fluid', 'qi_xu'),
    (lambda s: '血瘀' in s, 'qi_blood_fluid', 'xue_yu'),
    (lambda s: '血虚' in s, 'qi_blood_fluid', 'xue_xu'),
    (lambda s: '阴虚' in s, 'qi_blood_fluid', 'jin_ye_kui_xu'),
    (lambda s: '痰' in s, 'qi_blood_fluid', 'tan_zheng'),
    (lambda s: '湿' in s and '燥湿' not in s, 'patho_factors', 'shi'),
    (lambda s: '肝阳上亢' in s or '肝风' in s, 'zangfu', 'gan'),
    (lambda s: '肝风' in s or '中风' in s or '风痰' in s, 'patho_factors', 'feng'),
    (lambda s: '寒' in s and '寒热' not in s, 'patho_factors', 'han'),
    (lambda s: '火' in s or ('热' in s and '寒热' not in s), 'patho_factors', 'huo'),
]


def check_syndrome_conflict(syndromes: list, bz: dict) -> list[str]:
    conflicts = []
    for match_fn, section, key in SYNDROME_PARAM_RULES:
        if any(match_fn(s) for s in syndromes) and bz[section][key] == 0:
            conflicts.append(f"证型含相关特征但{section}.{key}=0")
    return conflicts


# ==========================================
# 断点续传
# ==========================================
def load_progress() -> dict:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


# ==========================================
# 统计修改
# ==========================================
def count_changes(old_bz: dict, new_bz: dict) -> int:
    changes = 0
    for section in ["bagang", "zangfu", "qi_blood_fluid", "patho_factors"]:
        for key in old_bz[section]:
            if old_bz[section][key] != new_bz[section][key]:
                changes += 1
    return changes


# ==========================================
# 处理单条记录
# ==========================================
def process_one(idx: int, record: dict) -> dict:
    record = fix_converted(record)
    c = record["converted"]

    if "four_diagnosis" not in c or "bianzheng" not in c:
        return record

    prompt = build_user_prompt(record)
    resp = call_api(prompt)
    if not resp:
        return record

    parsed = parse_response(resp)
    if not parsed:
        return record

    result = json.loads(json.dumps(record))

    if "zhongyi_wangwenqiezhen" in parsed:
        wqz = parsed["zhongyi_wangwenqiezhen"]
        for prefix in ["中医望闻切诊：", "中医望闻切诊:"]:
            if wqz.startswith(prefix):
                wqz = wqz[len(prefix):]
                break
        result["raw_data"]["中医望闻切诊"] = wqz

    if "four_diagnosis" in parsed:
        result["converted"]["four_diagnosis"] = validate_four_diagnosis(
            parsed["four_diagnosis"], record["converted"]["four_diagnosis"]
        )

    original_bz = record["converted"]["bianzheng"]
    new_bz = validate_bianzheng(parsed["bianzheng"], original_bz)
    result["converted"]["bianzheng"] = new_bz

    if "explanations" in parsed:
        result["converted"]["explanations"] = parsed["explanations"]

    conflicts = check_syndrome_conflict(record["converted"]["syndrome"], new_bz)
    result["_syndrome_conflict"] = 1 if conflicts else 0

    return result


# ==========================================
# 处理单个文件
# ==========================================
def process_file(input_name: str, output_name: str, progress: dict):
    input_path = os.path.join(DATA_DIR, input_name)
    output_path = os.path.join(DATA_DIR, output_name)

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    file_key = input_name
    done_indices = set(progress.get(file_key, []))

    todo = [(i, r) for i, r in enumerate(records) if i not in done_indices]
    print(f"\n{input_name}: 共 {len(records)} 条，已完成 {len(done_indices)}，剩余 {len(todo)}")

    if not todo:
        return 0, 0

    total_changes = 0
    changed_records = 0
    fd_filled = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(process_one, idx, record): (idx, record)
            for idx, record in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=input_name):
            idx, original = futures[future]
            try:
                verified = future.result()

                if "bianzheng" in original.get("converted", {}):
                    c = count_changes(original["converted"]["bianzheng"], verified["converted"]["bianzheng"])
                    if c > 0:
                        total_changes += c
                        changed_records += 1

                orig_fd = original.get("converted", {}).get("four_diagnosis", {})
                new_fd = verified.get("converted", {}).get("four_diagnosis", {})
                for k in ["wang", "wen", "wen_zhen", "qie"]:
                    if is_missing(orig_fd.get(k, "")) and not is_missing(new_fd.get(k, "")):
                        fd_filled += 1
                        break

                with write_lock:
                    done_indices.add(idx)
                    progress[file_key] = list(done_indices)
                    save_progress(progress)
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(verified, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  第{idx}条失败: {e}")
                with write_lock:
                    done_indices.add(idx)
                    progress[file_key] = list(done_indices)
                    save_progress(progress)
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(original, ensure_ascii=False) + "\n")

    print(f"{input_name} 完成: {changed_records}/{len(records)} 条辨证有修改，共 {total_changes} 个参数变更，{fd_filled} 条四诊补足")
    return changed_records, total_changes


# ==========================================
# 主流程
# ==========================================
def main():
    progress = load_progress()
    grand_changed = 0
    grand_total = 0

    for input_name, output_name in INPUT_FILES:
        changed, total = process_file(input_name, output_name, progress)
        grand_changed += changed
        grand_total += total

    print(f"\n全部完成！共 {grand_changed} 条记录辨证有修改，{grand_total} 个参数变更")


if __name__ == "__main__":
    main()
