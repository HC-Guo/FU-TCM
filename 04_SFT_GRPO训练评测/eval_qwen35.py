#!/usr/bin/env python3
"""Transformers 本地推理评测脚本：加载 SFT 后的 Qwen3.5 多模态模型测 TCM benchmark。"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PROJECT_DIR = Path(os.getenv("TCM_PROJECT_DIR", ROOT))

MODEL_PATH = os.getenv("TCM_MODEL_PATH", str(PROJECT_DIR / "saves" / "Qwen3.5-9B" / "coldstart_v1"))
BENCHMARK_DIR = Path(os.getenv("TCM_BENCHMARK_DIR", str(PROJECT_DIR / "benchmark")))

# ===== 选择要评测的 benchmark，仿照旧 eval_tcm.py 的写法 =====
BENCHMARK_PATH = Path(os.getenv("TCM_BENCHMARK_PATH", str(BENCHMARK_DIR / "tcm_benchmark.jsonl")))
# BENCHMARK_PATH = BENCHMARK_DIR / "tcm_benchmark_vision.jsonl"

IMAGE_DIRS = [
    BENCHMARK_DIR / "images",
]
BENCH_NAME = Path(BENCHMARK_PATH).stem
MODEL_NAME = f"{Path(MODEL_PATH).parent.name}_{Path(MODEL_PATH).name}"
EVAL_DIR = PROJECT_DIR / "eval_output" / BENCH_NAME / MODEL_NAME
OUTPUT_PATH = EVAL_DIR / "eval_results.json"
SUMMARY_PATH = EVAL_DIR / "eval_summary.txt"
REPORT_PATH = EVAL_DIR / "eval_report.html"
LETTERS = "ABCD"
NUM_RUNS = 1
SEED = 42
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.6
TOP_P = 0.9
ENABLE_THINKING = False
INFER_BATCH = 8  # 普通 Transformers 一次生成多少题；显存不够就改小。
RESUME = True  # 默认断点续写：读取 eval_responses_{run_id}.json，跳过已完成题目。


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_completed_json(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"skip broken response json: {path}")
        return {}
    return {row["uid"]: row for row in rows if row.get("uid")}


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def find_image(name: str | None) -> Path | None:
    if not name:
        return None
    p = Path(name)
    candidates = [p] if p.is_absolute() else [ROOT / name] + [d / name for d in IMAGE_DIRS]
    candidates += [d / p.name for d in IMAGE_DIRS]
    return next((x for x in candidates if x.exists()), None)


def make_prompt(row: dict) -> str:
    opts = "\n".join(f"{k}. {row.get('options', {}).get(k, '')}" for k in LETTERS)
    is_multi = row.get("question_type", "") == "多项选择题"
    if is_multi:
        return (
            "以下是一道中医多选题，请直接给出所有正确选项的字母，不需要解释。\n\n"
            f"{row.get('question', '')}\n{opts}\n\n答案："
        )
    return (
        "以下是一道中医单选题，请直接给出正确选项的字母，只给出一个选项。\n\n"
        f"{row.get('question', '')}\n{opts}\n\n答案："
    )


def normalize_letters(value, valid_options) -> str:
    valid = "".join(k for k in LETTERS if k in valid_options) or LETTERS
    if isinstance(value, (list, tuple, set)):
        parts = []
        for x in value:
            if isinstance(x, int) and 0 <= x < len(valid):
                parts.append(valid[x])
            else:
                parts.append(str(x))
        value = "".join(parts)
    text = (
        str(value or "")
        .upper()
        .replace("Ａ", "A")
        .replace("Ｂ", "B")
        .replace("Ｃ", "C")
        .replace("Ｄ", "D")
    )
    return "".join(k for k in valid if k in text)


def first_letter(value: str, valid_options) -> str:
    valid = "".join(k for k in LETTERS if k in valid_options) or LETTERS
    text = (
        str(value or "")
        .upper()
        .replace("Ａ", "A")
        .replace("Ｂ", "B")
        .replace("Ｃ", "C")
        .replace("Ｄ", "D")
    )
    return next((ch for ch in text if ch in valid), "")


def gold_answer(row: dict) -> str:
    raw = row.get("answer_idx", row.get("answer", ""))
    return normalize_letters(raw, row.get("options", {}))


def parse_answer(text: str, row: dict) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    options = row.get("options", {})
    s = (text or "").upper().replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ", "C").replace("Ｄ", "D")
    is_multi = row.get("question_type", "") == "多项选择题"
    for pat in [
        r"(?:答案|正确答案|选择|选项)\s*(?:是|为|:|：)?\s*([ABCD](?:\s*[,，、/和及]\s*[ABCD])*)",
        r"([ABCD](?:\s*[,，、/和及]\s*[ABCD])*)\s*(?:项|选项)\s*(?:正确|符合|最合适|最准确)",
        r"^\s*[\(（\[【]?\s*([ABCD](?:\s*[,，、/和及]\s*[ABCD])*)",
    ]:
        m = re.search(pat, s)
        if m:
            return normalize_letters(m.group(1), options)
    compact = re.sub(r"\s+", "", s)
    hits = [k for k, v in options.items() if v and re.sub(r"\s+", "", str(v)) in compact]
    if len(hits) == 1:
        return normalize_letters(hits[0], options)
    letters = normalize_letters(s, options)
    return letters if is_multi else first_letter(s, options)


def load_samples(limit: int) -> list[dict]:
    rows = []
    bench = Path(BENCHMARK_PATH).expanduser().resolve()
    ds = "vision" if "vision" in bench.name else "text"
    for i, row in enumerate(read_jsonl(bench), 1):
        row["_uid"], row["_dataset"], row["_index"] = f"{ds}-{i}", ds, i
        row["_category"] = row.get("category") or row.get("original_type") or row.get("q_type") or row.get("diag_type") or ds
        rows.append(row)
    return rows[:limit] if limit else rows


def processor_candidate_paths(model_path: str) -> list[str]:
    path = Path(model_path).expanduser()
    candidates: list[Path] = [path]

    for meta_name in ["adapter_config.json", "config.json"]:
        meta_path = path / meta_name
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ["base_model_name_or_path", "tokenizer_name_or_path"]:
            value = meta.get(key)
            if isinstance(value, str) and value:
                candidates.append(Path(value).expanduser())

    parts = path.parts
    if "saves" in parts:
        idx = parts.index("saves")
        if idx + 1 < len(parts):
            candidates.append(PROJECT_DIR / "models" / parts[idx + 1])

    candidates.extend(
        [
            PROJECT_DIR / "models" / path.name,
            PROJECT_DIR / "models" / path.parent.name,
            path.parent,
            PROJECT_DIR / "models" / "Qwen3.5-9B",
        ]
    )

    out = []
    seen = set()
    for item in candidates:
        text = str(item)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


class TransformersModel:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            from transformers import Qwen3_5ForConditionalGeneration
            model_cls = Qwen3_5ForConditionalGeneration
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText
                model_cls = AutoModelForImageTextToText
            except ImportError:
                model_cls = AutoModelForCausalLM

        self.torch = torch
        self.processor = self.load_processor(AutoProcessor, model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.device = getattr(self.model, "device", next(self.model.parameters()).device)

    @staticmethod
    def load_processor(auto_processor, model_path: str):
        tried = []
        last_error = None
        for path in processor_candidate_paths(model_path):
            if not path or path in tried:
                continue
            tried.append(path)
            try:
                return auto_processor.from_pretrained(path, trust_remote_code=True)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Processor 加载失败。SFT 保存目录通常没有 tokenizer/image_processor，"
            "脚本已自动尝试从 PROJECT_DIR/models 下找原始 Qwen3.5 基座模型目录。\n"
            f"已尝试: {tried}\n最后错误: {last_error}"
        ) from last_error

    def set_seed(self, seed: int) -> None:
        import random

        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def apply_chat_template(self, row: dict, has_image: bool) -> str:
        prompt = make_prompt(row)
        if has_image:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if ENABLE_THINKING and not text.endswith("<think>\n"):
            text += "<think>\n"
        return text

    def generate_batch(self, rows: list[dict], images: list[Path | None], seed: int) -> list[str]:
        from PIL import Image

        self.set_seed(seed)
        if any(img is None for img in images) and any(img is not None for img in images):
            out = []
            for row, img in zip(rows, images):
                out.extend(self.generate_batch([row], [img], seed))
            return out

        texts = [self.apply_chat_template(row, img is not None) for row, img in zip(rows, images)]
        opened_images = []
        if all(img is not None for img in images):
            opened_images = [Image.open(img).convert("RGB") for img in images if img is not None]
            inputs = self.processor(
                text=texts,
                images=opened_images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        outputs = [
            self.processor.decode(generated[i][input_len:], skip_special_tokens=True).strip()
            for i in range(len(rows))
        ]
        for image in opened_images:
            image.close()
        return outputs


def summarize(rows: list[dict]) -> dict:
    out = {"total": len(rows), "correct": sum(r["correct"] for r in rows)}
    out["accuracy"] = out["correct"] / out["total"] if out["total"] else 0
    out["by_dataset"] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        part = [r for r in rows if r["dataset"] == ds]
        ok = sum(r["correct"] for r in part)
        out["by_dataset"][ds] = {"total": len(part), "correct": ok, "accuracy": ok / len(part)}
    out["by_category"] = {}
    for cat in sorted({r["category"] for r in rows}):
        part = [r for r in rows if r["category"] == cat]
        ok = sum(r["correct"] for r in part)
        out["by_category"][cat] = {"total": len(part), "correct": ok, "accuracy": ok / len(part)}
    return out


def average_summaries(summaries: list[dict]) -> dict:
    avg = {
        "total": summaries[0]["total"] if summaries else 0,
        "each_run_accuracy": [s["accuracy"] for s in summaries],
        "average_accuracy": sum(s["accuracy"] for s in summaries) / len(summaries) if summaries else 0,
        "by_dataset": {},
        "by_category": {},
    }
    for group in ["by_dataset", "by_category"]:
        keys = sorted({k for s in summaries for k in s[group]})
        for key in keys:
            vals = [s[group][key]["accuracy"] for s in summaries if key in s[group]]
            total = next(s[group][key]["total"] for s in summaries if key in s[group])
            avg[group][key] = {
                "total": total,
                "each_run_accuracy": vals,
                "average_accuracy": sum(vals) / len(vals),
            }
    return avg


def write_report(path: Path, avg: dict, rows: list[dict]) -> None:
    ds_rows = "".join(
        f"<tr><td>{k}</td><td>{v['total']}</td><td>{v['average_accuracy']:.2%}</td><td>{', '.join(f'{x:.2%}' for x in v['each_run_accuracy'])}</td></tr>"
        for k, v in avg["by_dataset"].items()
    )
    cat_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['total']}</td><td>{v['average_accuracy']:.2%}</td><td>{', '.join(f'{x:.2%}' for x in v['each_run_accuracy'])}</td></tr>"
        for k, v in avg["by_category"].items()
    )
    wrong_rows = "".join(
        "<tr>"
        f"<td>{html.escape(r['uid'])}</td><td>{r['gold']}</td><td>{r['pred'] or '-'}</td>"
        f"<td>{html.escape(r['question'])}</td><td><pre>{html.escape(r.get('response', ''))}</pre></td>"
        "</tr>"
        for r in rows
        if not r["correct"]
    )
    atomic_write_text(
        path,
        f"""<!doctype html><meta charset="utf-8"><title>TCM Benchmark 评测报告</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;margin:32px;background:#f7f8f6}}main{{max-width:1100px;margin:auto}}table{{border-collapse:collapse;width:100%;background:white;margin:14px 0 28px}}td,th{{border:1px solid #dbe2db;padding:8px;text-align:left;vertical-align:top}}th{{background:#edf4f1}}.card{{display:inline-block;background:white;border:1px solid #dbe2db;border-radius:8px;padding:14px;margin:8px 8px 8px 0;min-width:150px}}.num{{font-size:26px;font-weight:800}}pre{{white-space:pre-wrap;margin:0;max-width:260px}}</style>
<main><h1>TCM Benchmark 评测报告</h1><p>模型：{html.escape(MODEL_PATH)}</p>
<div class="card">总题数<div class="num">{avg['total']}</div></div>
<div class="card">平均准确率<div class="num">{avg['average_accuracy']:.2%}</div></div>
<div class="card">运行次数<div class="num">{len(avg['each_run_accuracy'])}</div></div>
<h2>分数据集</h2><table><tr><th>数据集</th><th>题数</th><th>平均准确率</th><th>各次准确率</th></tr>{ds_rows}</table>
<h2>分类别</h2><table><tr><th>类别</th><th>题数</th><th>平均准确率</th><th>各次准确率</th></tr>{cat_rows}</table>
<h2>错误样例（最后一次运行）</h2><table><tr><th>ID</th><th>答案</th><th>预测</th><th>题目</th><th>输出</th></tr>{wrong_rows}</table></main>""",
    )


def main() -> None:
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.limit)
    model = TransformersModel(MODEL_PATH)
    print(f"loaded {len(samples)} samples")

    images = [find_image(row.get("image_abs_path") or row.get("image_path")) for row in samples]
    image_by_uid = {row["_uid"]: img for row, img in zip(samples, images)}
    all_summaries = []
    all_runs = []
    last_rows = []
    for run_id in range(1, NUM_RUNS + 1):
        seed = SEED + run_id - 1
        response_path = EVAL_DIR / f"eval_responses_{run_id}.json"
        completed = load_completed_json(response_path) if RESUME else {}
        remaining = [row for row in samples if row["_uid"] not in completed]
        print(f"\nrun {run_id}/{NUM_RUNS}, seed={seed}")
        print(f"responses: {response_path} ({len(completed)}/{len(samples)} done)")

        with tqdm(total=len(samples), initial=len(completed), desc=f"run {run_id}/{NUM_RUNS}", unit="题", ncols=100) as pbar:
            done = len(completed)
            if done:
                pbar.set_postfix(acc=f"{sum(r['correct'] for r in completed.values()) / done:.2%}", saved=done)

            for start in range(0, len(remaining), INFER_BATCH):
                batch = remaining[start:start + INFER_BATCH]
                batch_images = [image_by_uid[row["_uid"]] for row in batch]
                t0 = time.time()
                outputs = model.generate_batch(batch, batch_images, seed)
                latency = round((time.time() - t0) / max(len(batch), 1), 2)
                for row, img, output in zip(batch, batch_images, outputs):
                    pred = parse_answer(output, row)
                    gold = gold_answer(row)
                    rec = {
                        "uid": row["_uid"],
                        "dataset": row["_dataset"],
                        "category": row["_category"],
                        "index": row["_index"],
                        "question": row.get("question", ""),
                        "options": row.get("options", {}),
                        "gold": gold,
                        "pred": pred,
                        "response": output,
                        "correct": pred == gold,
                        "error": "",
                        "image": str(img or ""),
                        "latency": latency,
                    }
                    completed[rec["uid"]] = rec
                rows = [completed[row["_uid"]] for row in samples if row["_uid"] in completed]
                atomic_write_json(response_path, rows)
                done = len(completed)
                correct = sum(r["correct"] for r in completed.values())
                pbar.update(len(batch))
                pbar.set_postfix(acc=f"{correct / done:.2%}", saved=done)

        rows = [completed[row["_uid"]] for row in samples if row["_uid"] in completed]
        atomic_write_json(response_path, rows)
        run_summary = summarize(rows)
        all_summaries.append(run_summary)
        all_runs.append({"run_id": run_id, "seed": seed, "summary": run_summary, "responses": rows})
        last_rows = rows
        print(f"[{len(rows)}/{len(samples)}] acc={run_summary['accuracy']:.2%}")

    avg = average_summaries(all_summaries)
    summary_lines = []
    for name, stat in avg["by_dataset"].items():
        runs = ", ".join(f"{x:.2%}" for x in stat["each_run_accuracy"])
        summary_lines.append(f"{name}: {stat['average_accuracy']:.2%} (各次: {runs})")
    overall_runs = ", ".join(f"{x:.2%}" for x in avg["each_run_accuracy"])
    summary_lines.append(f"overall: {avg['average_accuracy']:.2%} (各次: {overall_runs})")

    result = {
        "config": {
            "model_path": MODEL_PATH,
            "benchmark_path": str(BENCHMARK_PATH),
            "image_dirs": [str(p) for p in IMAGE_DIRS],
            "num_runs": NUM_RUNS,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "enable_thinking": ENABLE_THINKING,
        },
        "average": avg,
        "runs": all_runs,
        "summary": summary_lines,
    }
    atomic_write_json(OUTPUT_PATH, result)
    atomic_write_text(SUMMARY_PATH, "\n".join(summary_lines) + "\n")
    write_report(REPORT_PATH, avg, last_rows)
    print(f"results: {OUTPUT_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
