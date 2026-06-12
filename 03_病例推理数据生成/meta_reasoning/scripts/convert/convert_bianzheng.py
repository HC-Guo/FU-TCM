"""
中医病案辨证结构化转换脚本
将 meta_reasoning.json 转换为包含四诊、八纲辨证、脏腑辨证等完整结构化数据

支持多个 API 对比测试：填好配置后运行即可
"""
import asyncio
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI

# ==========================================
# 1. 中转站配置
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL") or ""

# ==========================================
# 2. 运行配置
# ==========================================
INPUT_FILE = RAW_DIR / "meta_reasoning.json"
MAPPING_FILE = CONFIG_DIR / "mapping_table.json"
SAMPLE_COUNT = 10             # 0 = 全部799条，或填具体数字
CONCURRENCY = 3               # 并发数

# 要对比的模型列表（中转站支持的模型名）
DEFAULT_MODELS_TO_TEST = [
    "claude-sonnet-4-6",
    "deepseek-reasoner",
    # "gpt-4o",
    # "qwen3.5-397b-a17b",
]
MODELS_TO_TEST = [
    model.strip()
    for model in os.getenv("META_REASONING_MODELS", "").split(",")
    if model.strip()
] or DEFAULT_MODELS_TO_TEST

# ==========================================
# 3. Prompt 构建
# ==========================================
TARGET_SCHEMA = '''{
  "case_id": "TCM_V1_000001",
  "case_info": {
    "gender": "女",
    "age": 45,
    "chief_complaint": "失眠2月，伴头晕耳鸣、腰膝酸软、夜间口干"
  },
  "four_diagnosis": {
    "inspection": "舌红少苔，形体偏瘦。面色少华。",
    "auscultation_olfaction": "语音稍低，无明显异常气味。",
    "inquiry": "五心烦热，盗汗，失眠多梦，腰膝酸软，头晕耳鸣，夜间口干。纳可，二便调。",
    "palpation": "脉细数。"
  },
  "bianzheng": {
    "bagang": {
      "biao_li": "里",
      "han_re": "热",
      "xu_shi": "虚",
      "yin_yang": "阴虚"
    },
    "zangfu": {
      "gan": 2,
      "xin": 1,
      "pi": 0,
      "fei": 0,
      "shen": 3
    },
    "qi_blood_fluid": {
      "qi_xu": 0, "qi_xian": 0, "qi_tuo": 0,
      "qi_zhi": 0, "qi_ni": 0, "qi_bi": 0,
      "xue_xu": 1, "xue_yu": 0, "xue_re": 0, "xue_han": 0,
      "jin_ye_kui_xu": 2,
      "tan_zheng": 0, "yin_zheng": 0,
      "shui_ting_zheng": 0, "nei_shi_zheng": 0
    },
    "patho_factors": {
      "feng": 0, "han": 0, "shu": 0,
      "shi": 0, "zao": 1, "huo": 1
    }
  },
  "syndrome": ["肝肾阴虚证"],
  "treatment": {
    "principle": "滋补肝肾，养阴清热",
    "formula": "杞菊地黄丸加减",
    "herbs": ["熟地黄", "山药", "山茱萸", "枸杞子", "菊花", "茯苓", "泽泻", "牡丹皮"],
    "modifications": "加酸枣仁养心安神以治失眠，加女贞子、墨旱莲滋补肾阴以加强补阴之力。"
  },
  "explanations": {
    "bagang": {
      "biao_li": "本案症状以脏腑功能失调表现为主，未见表证特征，判为里证。",
      "han_re": "见五心烦热、夜间口干、舌红、脉细数，偏热象，判为热证。",
      "xu_shi": "以腰膝酸软、头晕耳鸣、舌少苔等虚损表现为主，判为虚证。",
      "yin_yang": "夜间口干、盗汗、舌红少苔、脉细数提示阴液不足，判为阴虚。"
    },
    "zangfu": {
      "gan": "头晕耳鸣可与肝肾不足相关，肝有明显相关。",
      "xin": "失眠多梦提示心神失养，心轻度相关。",
      "pi": "无明显脾虚表现，脾不相关。",
      "fei": "无明显肺系症状，肺不相关。",
      "shen": "腰膝酸软、耳鸣、盗汗、夜间口干均提示肾阴不足，肾为核心病位。"
    },
    "qi_blood_fluid": {
      "qi_xu": "无少气懒言、自汗等气虚表现，标为0。",
      "xue_xu": "形体偏瘦、失眠多梦、头晕提示可有轻度血虚，标为1。",
      "jin_ye_kui_xu": "夜间口干、舌红少苔提示津液亏虚明显，标为2。"
    },
    "patho_factors": {
      "zao": "夜间口干、舌少苔提示兼有轻度燥象，标为1。",
      "huo": "五心烦热、舌红、脉细数提示有轻度虚热内扰，标为1。"
    },
    "syndrome": "以腰膝酸软、头晕耳鸣、盗汗、夜间口干、舌红少苔、脉细数为主要依据，提示肝肾阴液不足、虚热内生，判断为肝肾阴虚证。",
    "treatment": "肝肾阴虚，虚热内扰，治以滋补肝肾、养阴清热。选杞菊地黄丸为主方，取六味地黄丸滋补肾阴之底，加枸杞、菊花清肝明目。加酸枣仁养心安神治失眠，女贞子、墨旱莲加强滋阴。"
  }
}'''

def build_system_prompt(mapping_table_str: str) -> str:
    return f"""你是一位资深中医辨证专家。你的任务是将中医病案记录转换为结构化的辨证数据。

## 任务说明

根据输入的病案信息（包括性别、年龄、主诉、症状、望闻切诊、病史、体格检查、辅助检查、疾病、证型等），输出结构化的辨证JSON。

## 输出格式

严格按以下JSON Schema输出，不要添加其他字段：

{TARGET_SCHEMA}

## 字段说明

### case_info
- gender: 直接取原数据性别
- age: 提取数字（如"78岁"→78）
- chief_complaint: 取原数据主诉，清理多余空格

### four_diagnosis（四诊拆分规则）
从原数据的"症状"、"中医望闻切诊"、"体格检查"中拆分：
- **inspection（望诊）**：舌象（舌质、舌苔、舌下络脉）、面色、形体、神态、精神状态、皮肤等可见信息
- **auscultation_olfaction（闻诊）**：语声、呼吸声、咳声、气味等
- **inquiry（问诊）**：患者自述症状（寒热、汗、头身、胸腹、饮食、睡眠、二便、疼痛、情志等）
- **palpation（切诊）**：脉象，以及腹诊等触诊信息

### bianzheng（辨证参数）
- **bagang（八纲）**：
  - biao_li: "表"/"里"/"半表半里"/"表里同病"/"未定"
  - han_re: "寒"/"热"/"寒热错杂"/"未定"
  - xu_shi: "虚"/"实"/"虚实夹杂"/"未定"
  - yin_yang: "阴虚"/"阳虚"/"阴盛"/"阳盛"/"阴阳两虚"/"未定"

- **zangfu（脏腑）**：0=无关, 1=轻度相关, 2=明显相关, 3=主导因素
- **qi_blood_fluid（气血津液）**：同上0-3评分
- **patho_factors（六淫病理因素）**：同上0-3评分

### syndrome
证型数组。原数据用"|"分隔的复合证型拆成数组元素。

### treatment（治法方药）
- **principle**: 治法（如"益气活血，化痰通络"），根据证型推导
- **formula**: 方剂名（根据证型和处方药物推断最接近的经典方或合方，如"补阳还五汤加减"、"四君子汤合桃红四物汤"，无法对应经典方则写"自拟XX方"）
- **herbs**: 药物数组，直接取原数据处方字段的药名，不编造剂量
- **modifications**: 加减说明，解释相对基础方增减了哪些药、为什么

### explanations
每个辨证判断的推理依据。要求：
- 必须引用具体的症状/体征作为证据
- 简洁明了，1-2句话
- qi_blood_fluid 和 patho_factors 中标为0的项可以省略不写

## 辨证映射参考表

以下是辨证时参考的证据-参数映射关系（用于辅助判断，不要机械套用，需结合临床整体分析）：

{mapping_table_str}

## 重要规则

1. 严格基于原始数据，不能凭空添加症状
2. 四诊信息缺失时如实记录"病案未明确记录"
3. 辨证参数必须与证型一致（如证型含"气虚"，则qi_xu不应为0）
4. explanations中标为0的气血津液和病理因素项可省略
5. 输出纯JSON，不要包含markdown标记或其他文字
"""


def build_user_prompt(record: dict) -> str:
    return f"请对以下病案进行辨证结构化转换，输出JSON：\n\n{json.dumps(record, ensure_ascii=False, indent=2)}"


# ==========================================
# 4. 异步处理
# ==========================================
async def process_one(client, model_name, record, sem, index, case_id):
    """处理单条记录"""
    async with sem:
        user_prompt = build_user_prompt(record)
        start = time.time()

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CACHE},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            clean = content.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(clean)

            # 注入 case_id
            parsed["case_id"] = case_id

            elapsed = time.time() - start
            return {
                "record_index": index + 1,
                "original_id": record.get("ID", "unknown"),
                "case_id": case_id,
                "result": parsed,
                "status": "success",
                "time_seconds": round(elapsed, 1),
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                "record_index": index + 1,
                "original_id": record.get("ID", "unknown"),
                "case_id": case_id,
                "result": {"raw": content if "content" in dir() else "", "error": str(e)},
                "status": "failed",
                "time_seconds": round(elapsed, 1),
            }


async def run_model(model_name: str, data: list):
    """对单个模型跑全部样例"""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    # 用模型名生成文件名（去掉特殊字符）
    safe_name = model_name.replace("/", "_").replace(".", "-")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / f"bianzheng_{safe_name}.jsonl"

    # 断点续写：读取已完成的 ID
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("status") == "success":
                            done_ids.add(r.get("original_id"))
                    except json.JSONDecodeError:
                        pass
        print(f"  [{model_name}] 已有 {len(done_ids)} 条成功记录，跳过")

    pending = [(i, rec) for i, rec in enumerate(data) if rec.get("ID") not in done_ids]
    if not pending:
        print(f"  [{model_name}] 全部已完成")
        return output_file

    print(f"  [{model_name}] 待处理 {len(pending)} 条")

    tasks = []
    for i, rec in pending:
        case_id = f"TCM_V1_{(i+1):06d}"
        tasks.append(process_one(client, model_name, rec, sem, i, case_id))

    success = 0
    failed = 0
    with open(output_file, "a", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result["status"] == "success":
                success += 1
                tag = "OK"
            else:
                failed += 1
                tag = "FAIL"
            print(
                f"  [{model_name}] {tag} ID={result['original_id']} "
                f"({result['time_seconds']}s)"
            )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print(f"  [{model_name}] 完成: {success} 成功, {failed} 失败 → {output_file}")
    return output_file


# ==========================================
# 5. 结果对比
# ==========================================
def compare_results(model_names: list, data: list):
    """对比不同模型的输出质量"""
    results = {}
    for name in model_names:
        safe = name.replace("/", "_").replace(".", "-")
        fpath = PROCESSED_DIR / f"bianzheng_{safe}.jsonl"
        if not os.path.exists(fpath):
            continue
        records = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("status") == "success":
                            records.append(r)
                    except json.JSONDecodeError:
                        pass
        results[name] = records

    if not results:
        print("\n无输出文件可对比")
        return

    print("\n" + "=" * 60)
    print("模型对比报告")
    print("=" * 60)

    for name, records in results.items():
        times = [r["time_seconds"] for r in records]
        avg_time = sum(times) / len(times) if times else 0

        complete = 0
        schema_issues = []
        for r in records:
            res = r.get("result", {})
            issues = []
            for field in ["case_info", "four_diagnosis", "bianzheng", "syndrome", "explanations"]:
                if field not in res:
                    issues.append(f"缺少{field}")
            bz = res.get("bianzheng", {})
            for sub in ["bagang", "zangfu", "qi_blood_fluid", "patho_factors"]:
                if sub not in bz:
                    issues.append(f"缺少bianzheng.{sub}")
            syn = res.get("syndrome")
            if syn is not None and not isinstance(syn, list):
                issues.append("syndrome不是数组")
            fd = res.get("four_diagnosis", {})
            for diag in ["inspection", "auscultation_olfaction", "inquiry", "palpation"]:
                if diag not in fd:
                    issues.append(f"缺少four_diagnosis.{diag}")
            if not issues:
                complete += 1
            else:
                schema_issues.append({"id": r.get("original_id"), "issues": issues})

        print(f"\n--- {name} ---")
        print(f"  成功数: {len(records)}")
        print(f"  平均耗时: {avg_time:.1f}s")
        if records:
            print(f"  Schema完整率: {complete}/{len(records)} ({complete/len(records)*100:.0f}%)")
        if schema_issues:
            print(f"  Schema问题:")
            for si in schema_issues[:3]:
                print(f"    ID={si['id']}: {', '.join(si['issues'])}")

    if len(results) >= 2:
        print("\n" + "-" * 60)
        print("逐条对比（第1条样例）")
        print("-" * 60)
        names = list(results.keys())
        id_sets = [set(r["original_id"] for r in recs) for recs in results.values()]
        common_ids = id_sets[0]
        for s in id_sets[1:]:
            common_ids &= s
        if common_ids:
            sample_id = sorted(common_ids)[0]
            for name in names:
                rec = next(r for r in results[name] if r["original_id"] == sample_id)
                res = rec.get("result", {})
                print(f"\n  [{name}] ID={sample_id}")
                bg = res.get("bianzheng", {}).get("bagang", {})
                print(f"    八纲: 表里={bg.get('biao_li')} 寒热={bg.get('han_re')} "
                      f"虚实={bg.get('xu_shi')} 阴阳={bg.get('yin_yang')}")
                zf = res.get("bianzheng", {}).get("zangfu", {})
                print(f"    脏腑: 心={zf.get('xin')} 肝={zf.get('gan')} "
                      f"脾={zf.get('pi')} 肺={zf.get('fei')} 肾={zf.get('shen')}")
                print(f"    证型: {res.get('syndrome')}")
                fd = res.get("four_diagnosis", {})
                print(f"    望诊: {fd.get('inspection', 'N/A')[:60]}...")
                print(f"    切诊: {fd.get('palpation', 'N/A')[:60]}...")


# ==========================================
# 6. 主函数
# ==========================================
SYSTEM_PROMPT_CACHE = ""

async def main():
    global SYSTEM_PROMPT_CACHE

    # 加载映射表
    print("加载映射表...")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping_str = json.dumps(mapping, ensure_ascii=False, indent=2)

    # 构建系统提示（全局缓存，避免重复构建）
    SYSTEM_PROMPT_CACHE = build_system_prompt(mapping_str)

    # 加载数据
    print("加载病案数据...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if SAMPLE_COUNT > 0:
        sample_data = all_data[:SAMPLE_COUNT]
        print(f"共 {len(all_data)} 条，取前 {SAMPLE_COUNT} 条测试")
    else:
        sample_data = all_data
        print(f"共 {len(all_data)} 条，全部处理")

    # 检查 API 配置
    if not API_KEY or not BASE_URL:
        print("\n错误: 请先设置 OPENAI_API_KEY/API_KEY 和 OPENAI_BASE_URL/API_BASE_URL。")
        return

    # 逐个模型运行
    print(f"\n开始测试 {len(MODELS_TO_TEST)} 个模型: {MODELS_TO_TEST}")
    for model in MODELS_TO_TEST:
        print(f"\n{'='*40}")
        print(f"运行模型: {model}")
        print(f"{'='*40}")
        await run_model(model, sample_data)

    # 对比结果
    compare_results(MODELS_TO_TEST, sample_data)


if __name__ == "__main__":
    asyncio.run(main())
