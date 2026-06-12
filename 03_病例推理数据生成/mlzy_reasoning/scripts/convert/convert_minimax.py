"""
中医病案辨证结构化转换脚本 — MiniMax API
将 train_set_599.json 转换为 prompt_example.json 格式的结构化辨证数据

核心原则：四诊 → 辨证参数 → 证型（正向推理，禁止从证型倒推）
"""
import asyncio
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI

# ==========================================
# 1. 配置区
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"

BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("MINIMAX_BASE_URL") or "https://api.minimax.chat/v1"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY") or os.getenv("API_KEY") or ""
MODEL_NAME = os.getenv("OPENAI_MODEL") or os.getenv("MINIMAX_MODEL") or "MiniMax-M2.7"

INPUT_FILE = PROCESSED_DIR / "名老中医_extracted.json"
OUTPUT_FILE = PROCESSED_DIR / "bianzheng_mlzy.jsonl"
MAPPING_FILE = CONFIG_DIR / "mapping_table.json"
EXAMPLE_FILE = PROJECT_ROOT / "prompt_example.json"

CONCURRENCY = 10
TEMPERATURE = 0.3

# ==========================================
# 2. 辨证核心规则摘要（提炼自中医诊断学/中医基础理论教材）
# ==========================================
TCM_TEXTBOOK_SUMMARY = """
## 辨证核心规则摘要（中医诊断学/中医基础理论）

### 一、八纲辨证

**表里辨证**
- 表证：新起病程短，恶寒发热并见，头身疼痛，鼻塞流涕，脉浮，苔薄白。病位在皮毛、肌腠、经络。
- 里证：病位在脏腑，症状以脏腑功能异常为主（胸闷、心悸、腹痛、腰酸等），脉沉、脉弦、脉细。
- 半表半里：往来寒热，胸胁苦满，口苦，咽干，脉弦。
- 判定要点：有表证特征则判表；无表证特征、症状指向脏腑则判里。

**寒热辨证**
- 寒证：畏寒肢冷，面色苍白，口淡不渴，小便清长，大便溏薄，舌淡苔白润，脉迟/紧。
- 热证：发热，面红目赤，口渴喜冷饮，烦躁不宁，小便短黄，大便干结，舌红苔黄，脉数。
- 注意：单独"苔白"（薄白苔）为正常舌象，不等于寒证，需有其他寒证证据支持。
- "脉数"为热证证据；"面色红润"为正常常色，不等于病理性面红。

**虚实辨证**
- 虚证：精神萎靡，面色无华，气短懒言，自汗盗汗，形体消瘦，脉虚弱无力。乏力为气虚证据。
- 实证：声高气粗，腹胀拒按，大便秘结，小便短赤，舌苔厚腻，脉实有力。痰浊、血瘀、食积属实。
- 虚实夹杂：正虚与邪实并存，如气虚+血瘀、阴虚+痰热。

**阴阳辨证**
- 阴证：面色苍白，精神萎靡，畏寒肢冷，气短懒言，脉沉迟无力。
- 阳证：面红目赤，烦躁不安，声高气粗，发热口渴，脉数有力。
- 阴虚：潮热盗汗，五心烦热，口干咽燥，舌红少苔，脉细数。
- 阳虚：畏寒肢冷，面色白，神疲嗜卧，脉沉迟。
- 判定要点：阴阳证据不足以确判时标"未定"。

### 二、脏腑辨证定位

**心**：心悸/心慌（中等证据_2）、胸闷/胸痛（中等证据_2）、失眠（弱证据_1）、健忘、心烦。
**肝**：胁痛/胁胀（中等证据_2）、目赤（中等证据_2）、急躁易怒、头晕目眩（弱证据_1）、口苦。
**脾**：纳差/食欲不振（中等证据_2）、腹胀（中等证据_2）、便溏/大便稀（中等证据_2）、乏力（弱证据_1）、面色萎黄。
**肺**：咳嗽（中等证据_2）、气喘/憋喘（中等证据_2）、气短（弱证据_1）、鼻塞。
**肾**：腰膝酸软（中等证据_2）、耳鸣/听力下降（弱证据_1）、夜尿频多、水肿（下肢为主）。

### 三、气血津液辨证

**气虚**：少气懒言（强证据_3）、自汗（强证据_3）、乏力/神疲（中等证据_2）、脉虚弱。
**气陷**：脏器下垂、小腹坠胀、久泻脱肛。
**气脱**：大汗淋漓、四肢厥冷、呼吸微弱（危象）。
**气滞**：胸胁胀痛、嗳气、矢气后舒。
**气逆**：呕吐、呃逆、咳嗽气喘（肺气上逆）。
**气闭**：突然昏倒、牙关紧闭、不省人事。
**血虚**：面色苍白无华（强证据_3）、唇甲色淡、头晕眼花、心悸失眠、脉细。
**血瘀**：舌暗/舌紫（中等证据_2）、舌下络脉迂曲（中等证据_2）、固定刺痛、肌肤甲错。
**血热**：吐血、衄血、皮肤斑疹、舌红绛、脉数。
**血寒**：肢冷、经色紫暗有块、少腹冷痛。
**津液亏虚**：口干、咽干（中等证据_2）、皮肤干燥、小便短少、大便干结、舌红少津。
**痰证**：痰多、苔腻（中等证据_2）、胸闷、肢体沉重。
**饮证**：胸胁积液、心下悸、咳嗽吐清稀痰。
**水停证**：水肿、小便不利、腹水。
**内湿证**：肢体沉重、口黏、苔腻、大便黏滞。

### 四、六淫病因辨证

**风**：游走性疼痛、肢体抽搐、皮肤瘙痒、脉浮。善行数变。
**寒**：畏寒肢冷、冷痛喜温、面色苍白、脉迟紧。注意：单独苔白不构成寒邪证据。
**暑**：夏季发病、高热大汗、烦渴引饮、气短乏力。
**湿**：身重如裹、肢体沉重、苔腻、口黏、大便黏滞、脉濡。
**燥**：口鼻干燥、皮肤干裂、干咳少痰、大便干结。
**火（热）**：口渴喜冷饮、面红目赤、烦躁、苔黄、脉数。注意：单独脉数为弱证据_1。

### 五、重要判定原则

1. **正向推理**：必须从四诊信息出发推导辨证参数，再由参数组合判断证型。严禁从证型名称反推参数。
2. **证据分级**：0=无关，1=弱证据，2=中等证据，3=强/主导证据。多个弱证据可升级。
3. **正常体征不作为证据**："面色红润"为正常常色，"薄白苔"为正常舌象。
4. **标为0的项必须有合理解释**：说明为何该项无相关证据。
5. **syndrome直接取原数据证型**：用"|"分隔的拆为数组。
"""

# ==========================================
# 3. Prompt 构建
# ==========================================
TARGET_SCHEMA = """{
  "case_id": "TCM_V1_XXXXXX",
  "case_info": {
    "gender": "男/女",
    "age": 62,
    "chief_complaint": "主诉摘要（去掉句号和多余空格）"
  },
  "four_diagnosis": {
    "wang": "望诊：舌象、面色、形体、神态等可见信息",
    "wen": "闻诊：语声、呼吸声、气味等",
    "wen_zhen": "问诊：患者自述症状（寒热汗头身胸腹饮食睡眠二便等）",
    "qie": "切诊：脉象及触诊信息"
  },
  "bianzheng": {
    "bagang": {
      "biao_li": "表/里/半表半里/表里同病/未定",
      "han_re": "寒/热/寒热错杂/未定",
      "xu_shi": "虚/实/虚实夹杂/未定",
      "yin_yang": "阴/阳/阴阳并见/未定"
    },
    "zangfu": {
      "xin": 0, "gan": 0, "pi": 0, "fei": 0, "shen": 0
    },
    "qi_blood_fluid": {
      "qi_xu": 0, "qi_xian": 0, "qi_tuo": 0,
      "qi_zhi": 0, "qi_ni": 0, "qi_bi": 0,
      "xue_xu": 0, "xue_yu": 0, "xue_re": 0, "xue_han": 0,
      "jin_ye_kui_xu": 0,
      "tan_zheng": 0, "yin_zheng": 0,
      "shui_ting_zheng": 0, "nei_shi_zheng": 0
    },
    "patho_factors": {
      "feng": 0, "han": 0, "shu": 0,
      "shi": 0, "zao": 0, "huo": 0
    }
  },
  "syndrome": ["证型1", "证型2"],
  "explanations": {
    "bagang": {
      "biao_li": "引用具体症状/体征说明判断依据",
      "han_re": "...",
      "xu_shi": "...",
      "yin_yang": "..."
    },
    "zangfu": {
      "xin": "引用症状说明评分依据（标为0的也要简要说明）",
      "gan": "...", "pi": "...", "fei": "...", "shen": "..."
    },
    "qi_blood_fluid": {
      "qi_xu": "引用症状说明评分依据",
      "...（所有15个字段都要有解释）": "..."
    },
    "patho_factors": {
      "feng": "...", "han": "...", "shu": "...",
      "shi": "...", "zao": "...", "huo": "..."
    },
    "syndrome": {
      "证型名1": "基于四诊症状的综合推理，说明为何判定此证型",
      "证型名2": "..."
    }
  }
}"""


def build_system_prompt(mapping_table_str: str) -> str:
    return f"""你是一位资深中医辨证专家。你的任务是将中医病案记录转换为结构化的辨证数据。

## 核心原则（最高优先级）

**严格正向推理：四诊 → 辨证参数 → 证型**

1. 所有辨证参数（八纲、脏腑、气血津液、病理因素）必须且只能从四诊信息（望闻问切）中推导
2. **绝对禁止从证型名称反推辨证参数**。例如，不能因为证型是"阴虚阳亢证"就给肝、肾加分——必须有具体的四诊症状证据
3. syndrome（证型）字段直接取原数据的"证型"字段，用"|"分隔的拆为数组
4. explanations 中的每个解释必须引用具体的四诊症状/体征作为证据，不得出现"因证型涉及XX"、"结合证型推导"等表述

## 输出格式

严格按以下JSON Schema输出，不要添加其他字段：

{TARGET_SCHEMA}

## 字段说明

### case_id
格式：TCM_V1_ + 原始ID补零到6位（如ID="25" → "TCM_V1_000025"）

### case_info
- gender: 直接取原数据性别
- age: 提取数字（如"78岁"→78，"62"→62）
- chief_complaint: 取原数据主诉，去掉"主  诉："前缀、句号和多余空格

### four_diagnosis（四诊拆分规则）
从原数据的"症状"和"中医望闻切诊"中拆分：
- **wang（望诊）**：舌象（舌质、舌苔、舌下络脉）、面色、形体、神态、表情等可见信息
- **wen（闻诊）**：语声、呼吸声、咳声、气味等
- **wen_zhen（问诊）**：患者自述症状（从"症状"字段提取）
- **qie（切诊）**：脉象信息

### bianzheng（辨证参数）
- **bagang（八纲）**：从给定选项中选择
- **zangfu（脏腑）**：0=无关, 1=弱证据/轻度相关, 2=中等证据/明显相关, 3=强证据/主导因素
- **qi_blood_fluid（气血津液）**：同上0-3评分
- **patho_factors（六淫病理因素）**：同上0-3评分

### syndrome
证型数组。原数据"证型"字段用"|"分隔的拆成数组元素。

### explanations
每个辨证判断的推理依据：
- 必须引用具体的四诊症状/体征作为证据
- 简洁明了，1-2句话
- 所有字段都要有解释，包括标为0的项（简要说明无相关证据即可）
- syndrome 下的每个证型解释：基于四诊症状综合推理

## 辨证核心规则（中医教材摘要）

{TCM_TEXTBOOK_SUMMARY}

## 辨证映射参考表

以下是辨证时参考的证据-参数映射关系（用于辅助判断，不要机械套用，需结合临床整体分析）：

{mapping_table_str}

## 重要规则

1. 严格基于原始数据的四诊信息，不能凭空添加症状
2. 四诊信息中未明确提及的项，在拆分时如实记录
3. 正常体征不作为病理证据（如"面色红润"为正常常色，"薄白苔"为正常舌象）
4. 辨证参数只能从四诊症状推导，不能从证型名称反推
5. 输出纯JSON，不要包含markdown标记或其他文字
"""


def build_user_prompt(example_raw: dict, example_converted: dict, record: dict) -> str:
    raw_str = json.dumps(example_raw, ensure_ascii=False, indent=2)
    conv_str = json.dumps(example_converted, ensure_ascii=False, indent=2)
    rec_str = json.dumps(record, ensure_ascii=False, indent=2)
    return f"""以下是一个完整的转换示例，请严格参照其格式、推理深度和正向推理原则：

【示例】
输入：
{raw_str}

输出：
{conv_str}

---

现在请对以下病案进行辨证结构化转换，输出JSON（只输出与上面"输出"相同结构的JSON，包含case_id、case_info、four_diagnosis、bianzheng、syndrome、explanations）：

{rec_str}"""


# ==========================================
# 4. 异步处理
# ==========================================
async def process_one(client, system_prompt, example_raw, example_converted, record, sem, index):
    """处理单条记录"""
    async with sem:
        raw_id = record.get('ID', str(index))
        # 兼容纯数字ID和MLZ_前缀ID
        if isinstance(raw_id, str) and raw_id.startswith('MLZ_'):
            case_id = f"TCM_V2_{raw_id[4:]}"
        else:
            case_id = f"TCM_V1_{int(raw_id):06d}"
        user_prompt = build_user_prompt(example_raw, example_converted, record)
        start = time.time()

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content
            # MiniMax-M2.7 returns <think>...</think> before JSON
            clean = content
            if "</think>" in clean:
                clean = clean.split("</think>", 1)[1]
            clean = clean.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(clean)

            # 确保 case_id 正确
            parsed["case_id"] = case_id

            elapsed = time.time() - start
            return {
                "record_index": index + 1,
                "original_id": record.get("ID", "unknown"),
                "case_id": case_id,
                "raw_data": record,
                "converted": parsed,
                "status": "success",
                "time_seconds": round(elapsed, 1),
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                "record_index": index + 1,
                "original_id": record.get("ID", "unknown"),
                "case_id": case_id,
                "raw_data": record,
                "converted": {"raw": content if "content" in dir() else "", "error": str(e)},
                "status": "failed",
                "time_seconds": round(elapsed, 1),
            }


def auto_clean_failed_records(filepath):
    """启动时剔除 failed 记录"""
    if not os.path.exists(filepath):
        return
    valid_lines = []
    failed_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    valid_lines.append(line)
                else:
                    failed_count += 1
            except json.JSONDecodeError:
                failed_count += 1
    if failed_count > 0:
        print(f"[auto clean] removed {failed_count} failed/corrupt records")
        with open(filepath, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line)


# ==========================================
# 5. 主函数
# ==========================================
async def main():
    # 检查配置
    if not API_KEY or not BASE_URL:
        print("Error: please set OPENAI_API_KEY/MINIMAX_API_KEY/API_KEY and BASE_URL first!")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    auto_clean_failed_records(OUTPUT_FILE)

    # 加载映射表
    print("loading mapping table...")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping_str = json.dumps(mapping, ensure_ascii=False, indent=2)

    # 加载示例
    print("loading example...")
    with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
        example_data = json.load(f)
    example_raw = example_data["raw_data"]
    example_converted = example_data["converted"]

    # 构建 system prompt
    system_prompt = build_system_prompt(mapping_str)
    print(f"system prompt length: {len(system_prompt)} chars")

    # 加载训练数据
    print(f"loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"total records: {len(all_data)}")

    # 断点续写
    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("status") == "success":
                            done_ids.add(r.get("original_id"))
                    except json.JSONDecodeError:
                        pass
        if done_ids:
            print(f"found {len(done_ids)} completed records, skipping")

    pending = [(i, rec) for i, rec in enumerate(all_data) if rec.get("ID") not in done_ids]
    if not pending:
        print("all records done!")
        return

    print(f"pending: {len(pending)} records")
    print(f"model: {MODEL_NAME}, concurrency: {CONCURRENCY}, temperature: {TEMPERATURE}")
    print()

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = []
    for i, rec in pending:
        tasks.append(process_one(client, system_prompt, example_raw, example_converted, rec, sem, i))

    success = 0
    failed = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result["status"] == "success":
                success += 1
                tag = "OK"
            else:
                failed += 1
                tag = "FAIL"
            print(
                f"[{tag}] #{result['record_index']} ID={result['original_id']} "
                f"({result['time_seconds']}s)  [{success+failed}/{len(pending)}]"
            )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nDone: {success} success, {failed} failed -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
