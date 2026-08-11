"""
将本地 train/test JSONL 转为 verl 要求的 parquet 格式。
字段：data_source, prompt, ability, reward_model, extra_info
"""
import json
import os
from pathlib import Path
import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = Path(
    os.getenv("TCM_GRPO_TRAIN_FILE", str(PROCESSED_DIR / "bianzheng_mlzy_train.jsonl"))
).expanduser()
TEST_FILE = Path(
    os.getenv("TCM_GRPO_TEST_FILE", str(PROCESSED_DIR / "bianzheng_mlzy_test.jsonl"))
).expanduser()
EXAMPLE_FILE = Path(
    os.getenv("TCM_PROMPT_EXAMPLE_FILE", str(PROJECT_ROOT / "prompt_example.json"))
).expanduser()
OUTPUT_DIR = Path(
    os.getenv("TCM_VERL_OUTPUT_DIR", str(PROJECT_ROOT / "data_parquet"))
).expanduser()

# 加载 few-shot 示例（启动时构建一次）
EXAMPLE_STR = None

def _load_example():
    global EXAMPLE_STR
    if EXAMPLE_STR is not None:
        return EXAMPLE_STR

    if not EXAMPLE_FILE.exists():
        EXAMPLE_STR = ""
        return EXAMPLE_STR

    with EXAMPLE_FILE.open("r", encoding="utf-8") as f:
        ex = json.load(f)

    ex_raw = ex["raw_data"]
    ex_input = {
        "ID": ex_raw.get("ID", ""),
        "性别": ex_raw.get("性别", ""),
        "年龄": ex_raw.get("年龄", ""),
        "主诉": ex_raw.get("主诉", ""),
        "症状": ex_raw.get("症状", ""),
        "中医望闻切诊": ex_raw.get("中医望闻切诊", ""),
    }

    fd = ex["converted"]["four_diagnosis"]
    expl = ex["converted"]["explanations"]

    bz = ex["converted"]["bianzheng"]

    think_lines = [
        "<think>",
        "一、四诊提取：",
        f"望诊：{fd['wang']}",
        f"闻诊：{fd['wen']}",
        f"问诊：{fd['wen_zhen']}",
        f"切诊：{fd['qie']}",
        "",
        "二、辨证参数推导：",
        "【八纲】",
    ]
    for key, name in [("biao_li", "表里"), ("han_re", "寒热"), ("xu_shi", "虚实"), ("yin_yang", "阴阳")]:
        think_lines.append(f"{name}：{bz['bagang'][key]}。{expl['bagang'][key]}")
    think_lines.append("【脏腑】")
    for key, name in [("xin", "心"), ("gan", "肝"), ("pi", "脾"), ("fei", "肺"), ("shen", "肾")]:
        think_lines.append(f"{name}={bz['zangfu'][key]}：{expl['zangfu'][key]}")
    think_lines.append("【气血津液】")
    for key, name in [
        ("qi_xu", "气虚"), ("qi_xian", "气陷"), ("qi_tuo", "气脱"),
        ("qi_zhi", "气滞"), ("qi_ni", "气逆"), ("qi_bi", "气闭"),
        ("xue_xu", "血虚"), ("xue_yu", "血瘀"), ("xue_re", "血热"), ("xue_han", "血寒"),
        ("jin_ye_kui_xu", "津液亏虚"), ("tan_zheng", "痰证"), ("yin_zheng", "饮证"),
        ("shui_ting_zheng", "水停证"), ("nei_shi_zheng", "内湿证"),
    ]:
        think_lines.append(f"{name}={bz['qi_blood_fluid'][key]}：{expl['qi_blood_fluid'][key]}")
    think_lines.append("【六淫】")
    for key, name in [("feng", "风"), ("han", "寒"), ("shu", "暑"), ("shi", "湿"), ("zao", "燥"), ("huo", "火")]:
        think_lines.append(f"{name}={bz['patho_factors'][key]}：{expl['patho_factors'][key]}")
    think_lines.append("")
    think_lines.append("三、证型判定：")
    for syn_name, syn_expl in expl["syndrome"].items():
        think_lines.append(f"{syn_name}：{syn_expl}")
    think_lines.append("</think>")

    # 输出 JSON 只保留证型
    ex_output_json = json.dumps(ex["converted"]["syndrome"], ensure_ascii=False)
    ex_output = "\n".join(think_lines) + "\n" + ex_output_json

    EXAMPLE_STR = (
        f"【示例】\n"
        f"输入：\n{json.dumps(ex_input, ensure_ascii=False, indent=2)}\n\n"
        f"输出：\n{ex_output}"
    )
    return EXAMPLE_STR


def build_prompt(raw_data: dict) -> list[dict]:
    """构造 chat_template 格式的 prompt：单条 user 消息，含示例 + 当前病案"""
    example = _load_example()
    input_fields = {
        "ID": raw_data.get("ID", ""),
        "性别": raw_data.get("性别", ""),
        "年龄": raw_data.get("年龄", ""),
        "主诉": raw_data.get("主诉", ""),
        "症状": raw_data.get("症状", ""),
        "中医望闻切诊": raw_data.get("中医望闻切诊", ""),
    }

    example_section = f"{example}\n\n---\n\n" if example else ""
    content = (
        "你是一位经验丰富的国医大师，请根据以下病案信息进行辨证分析，判断证型。\n"
        "要求：\n"
        "1. 先在 <think>...</think> 中进行推理，包含：四诊提取、辨证参数推导（八纲/脏腑/气血津液/六淫）、证型判定三个步骤\n"
        "2. 然后输出证型JSON数组，如 [\"证型1\", \"证型2\"]\n\n"
        f"{example_section}"
        f"现在请对以下病案进行辨证分析：\n\n"
        f"病案：\n{json.dumps(input_fields, ensure_ascii=False, indent=2)}"
    )

    return [{"role": "user", "content": content}]


def load_records(path: Path) -> list[dict]:
    """Load JSON or JSONL records and retain successful conversions."""
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("status", "success") == "success"]


def process_train():
    data = load_records(INPUT_FILE)
    if not data:
        raise FileNotFoundError(f"No training records found: {INPUT_FILE}")

    records = []
    for idx, item in enumerate(data):
        raw = item["raw_data"]
        converted = item["converted"]

        records.append({
            "data_source": "tcm_bianzheng",
            "prompt": build_prompt(raw),
            "ability": "tcm",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(converted, ensure_ascii=False),
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "original_id": raw.get("ID", ""),
            },
        })

    ds = datasets.Dataset.from_list(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "train.parquet"
    ds.to_parquet(str(out_path))
    print(f"Train: {len(records)} records -> {out_path}")


def process_test():
    if not TEST_FILE.exists():
        print(f"Test file not found: {TEST_FILE}, skipping")
        return

    data = load_records(TEST_FILE)

    records = []
    for idx, item in enumerate(data):
        raw = item["raw_data"]
        converted = item["converted"]

        records.append({
            "data_source": "tcm_bianzheng",
            "prompt": build_prompt(raw),
            "ability": "tcm",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(converted, ensure_ascii=False),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "original_id": raw.get("ID", ""),
            },
        })

    ds = datasets.Dataset.from_list(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test.parquet"
    ds.to_parquet(str(out_path))
    print(f"Test: {len(records)} records -> {out_path}")


if __name__ == "__main__":
    process_train()
    process_test()
