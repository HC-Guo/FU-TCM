from __future__ import annotations
import os, sys, json, subprocess, re
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_run_dataflow_dir() -> Path:
    candidates = [
        PROJECT_ROOT.parent / "run_dataflow",
        PROJECT_ROOT.parent / "Dataflow" / "run_dataflow",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_path_from_env(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    return Path(raw).expanduser() if raw else default


RUNDATAFLOW_DIR = _resolve_run_dataflow_dir()
WORK_DIR = PROJECT_ROOT / "shezhen_vqa_workdir"
CACHE_DIR = _resolve_path_from_env("SHEZHEN_VQA_CACHE_DIR", WORK_DIR / "cache")
OUTPUT_DIR = _resolve_path_from_env("SHEZHEN_VQA_OUTPUT_DIR", WORK_DIR / "output")

sys.path.insert(0, str(RUNDATAFLOW_DIR / "api_pipelines"))
sys.path.insert(0, str(RUNDATAFLOW_DIR))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(str(WORK_DIR))

# ── 导入 ──────────────────────────────────────────────────────────────────────
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger, PDF_Merger
from dataflow.pipeline import PipelineABC
from chapter_based_generator import ChapterBasedGenerator
from zhaoliming_output_parser import ZhaoLiMingOutputParser


# 全局 mineru_output 目录（step1 转换的结果存在这里）
GLOBAL_MINERU_OUT = WORK_DIR / "mineru_output"


def run_local_mineru(pdf_path: str, out_dir: str, backend: str = "vlm-auto-engine", book_name: str = "") -> str:
    """
    调用本地 mineru 命令行转换 PDF，返回生成的 .md 文件路径。
    优先使用 step1 已转换好的全局目录，不存在才重新转换。
    """
    pdf = Path(pdf_path)

    # 1. 优先查找 step1 全局 mineru_output（shezhen_vqa_workdir/mineru_output/书名/）
    if book_name:
        global_book_dir = GLOBAL_MINERU_OUT / book_name
        if global_book_dir.exists():
            found = list(global_book_dir.rglob("*.md"))
            if found:
                print(f"  [跳过 MinerU] 使用已有结果: {found[0]}")
                return str(found[0])

    # 2. 查找 cache 目录下的结果
    expected_md = Path(out_dir) / pdf.stem / backend / f"{pdf.stem}.md"
    if expected_md.exists():
        print(f"  [跳过 MinerU] 已存在: {expected_md}")
        return str(expected_md)
    found = list(Path(out_dir).rglob("*.md")) if Path(out_dir).exists() else []
    if found:
        print(f"  [跳过 MinerU] 已存在: {found[0]}")
        return str(found[0])

    # 3. 重新转换
    print(f"  [MinerU] 转换: {pdf.name} ...")
    cmd = ["mineru", "-p", str(pdf), "-o", str(out_dir), "-b", backend, "--source", "local"]
    subprocess.run(cmd, check=True)

    if expected_md.exists():
        return str(expected_md)
    found = list(Path(out_dir).rglob(f"{pdf.stem}.md"))
    if found:
        return str(found[0])
    raise RuntimeError(f"MinerU 转换完成但未找到 .md 文件: {out_dir}")

# ── Prompt ────────────────────────────────────────────────────────────────────
SHEZHEN_VQA_PROMPT = (
    "你是一个中医舌诊图谱分析助手。你的任务是从中医舌诊带图书籍中提取视觉问答（VQA）对。\n\n"
    "输入是一个JSON列表，每个元素包含：\n"
    "- type: 类型（text文本, image图片, header标题, page_number页码）\n"
    "- id: 唯一标识符\n"
    "- text: 文本内容（type=text/header/page_number时）\n"
    "- img_path: 图片路径（type=image时）\n"
    "- image_caption: 图片标题（type=image时），如 '图1 淡红舌'、'图2-3 淡白舌' 等\n\n"
    "重要提示：本书的图文经常不在同一页！\n"
    "- 图片可能出现在一页，而对应的详细描述文字在相邻的上一页或下一页\n"
    "- 你需要综合前后多个页面的信息来理解图片内容\n\n"
    "任务要求：\n"
    "1. 遍历所有图片（type=\"image\"）\n"
    "2. 从 image_caption 中提取图号（如 '图1'、'图2-3' 等）\n"
    "3. 为每张图片生成2-4个问答对，覆盖：\n"
    "   - 舌象特征（颜色、形态、苔质等）\n"
    "   - 临床意义（主病、辨证）\n"
    "   - 相关病因病机（如有）\n"
    "   - 治疗原则（如有）\n\n"
    "处理多图共享文本：分别为每张图生成问答对，答案只描述当前图片内容。\n\n"
    "输出格式（XML）：\n"
    "<chapter>\n"
    "<title>章节标题ID</title>\n"
    "<qa_pair>\n"
    "<label>图片编号（如'图1'、'图2-3'）</label>\n"
    "<question>问题文本</question>\n"
    "<answer>答案文本</answer>\n"
    "</qa_pair>\n"
    "</chapter>\n\n"
    "规则：\n"
    "1. 每个<qa_pair>必须包含<label>, <question>, <answer>\n"
    "2. <label>填写图片的完整编号（保留'图'字）\n"
    "3. <question>直接针对图片内容提问\n"
    "4. <answer>要源于原文信息，准确描述舌象及其临床意义\n"
    "5. 章节标题选择最相关的header的id\n"
    "6. 每张图生成2-4个问答对\n"
    "7. 如果没有足够信息生成问答对，输出<empty></empty>\n"
    "8. 严禁输出任何思考过程、解释说明、代码块标记（如```xml/```）\n"
    "9. 最终输出必须且只能包含XML标签内容（<chapter>...</chapter> 或 <empty></empty>）\n"
)

# ── 目标书籍列表 ──────────────────────────────────────────────────────────────
SHEZHEN_BOOKS_DIR = PROJECT_ROOT / "舌诊带图书籍"

TARGET_BOOKS = [
    {"name": "中医舌诊彩色图谱_龚一萍",
     "pdf": "中医舌诊彩色图谱 中英文对照 (龚一萍) (Z-Library).pdf"},
    {"name": "中医舌诊彩色图谱_许家佗",
     "pdf": "中医舌诊彩色图谱：汉英对照（Color Atlas of Chinese Medical Tongue Diagnosis：Chinese-English Edition） (许家佗 主编； 费兆馥 主审) (Z-Library).pdf"},
    {"name": "中医舌诊临床图解_许家佗",
     "pdf": "中医舌诊临床图解 (许家佗) (Z-Library).pdf"},
    {"name": "中医舌诊完全图解_吴中朝",
     "pdf": "中医舌诊完全图解 (吴中朝, 王彤) (Z-Library).pdf"},
    {"name": "实用中医舌诊彩色图谱_宋天彬",
     "pdf": "实用中医舌诊彩色图谱 (宋天彬) (Z-Library).pdf"},
    {"name": "舌诊图谱_臧俊岐",
     "pdf": "舌诊图谱 (Pdg2Pic, 臧俊岐主编) (Z-Library).pdf"},
    {"name": "舌诊辨证图谱_周幸来",
     "pdf": "舌诊辨证图谱第2版 (周幸来，周举，浙江省江山市幸来特色医学研究所主编, 周幸来, 周举主编 etc.) (Z-Library).pdf"},
    {"name": "望舌诊疗图解_戴豪良",
     "pdf": "望舌诊疗图解 (戴豪良) (Z-Library).pdf"},
    {"name": "图解望舌诊病_陈家旭",
     "pdf": "图解望舌诊病 (陈家旭，刘晓明编著, 陈家旭, 刘晓明编著, 刘晓明, Liu xiao ming, 陈家旭) (Z-Library).pdf"},
    {"name": "温病舌诊图谱_张之文",
     "pdf": "温病舌诊图谱 第2版 (张之文编著, 张之文, 刘碧清主编, 刘碧清, Liu bi qing, 张之文) (Z-Library).pdf"},
    {"name": "中华舌诊观止_陆小左",
     "pdf": "中华舌诊观止 (陆小左，刘毅主编, 陆小左, 刘毅主编, 陆小左, 刘毅) (Z-Library).pdf"},
    {"name": "舌下络脉诊法图谱_袁红霞",
     "pdf": "舌下络脉诊法图谱 (袁红霞,袁和平主编) (Z-Library).pdf"},
    {"name": "形色舌诊_阎金海",
     "pdf": "形色舌诊 (阎金海，赵冀生编著, Yan jin hai., Zhao ji sheng, 阎金海 etc.) (Z-Library).pdf"},
    {"name": "中国舌诊大全",
     "pdf": "中国舌诊大全 (中国舌诊大全編輯委員會) (Z-Library).pdf"},
]


# ── Pipeline ──────────────────────────────────────────────────────────────────
class ShezhenVQAPipeline(PipelineABC):
    def __init__(
        self,
        book_name: str,
        skip_llm_if_cached: bool = True,
        sample_images: int | None = None,
        sample_window: int = 20,
        print_only: bool = False,
        debug_chunks: int = 1,
        archive_output_dir: Path | None = None,
    ):
        super().__init__()
        self.book_name = book_name
        self.skip_llm_if_cached = skip_llm_if_cached
        self.sample_images = sample_images
        self.sample_window = sample_window
        self.print_only = print_only
        self.debug_chunks = debug_chunks
        self.archive_output_dir = archive_output_dir or OUTPUT_DIR
        self.archive_output_dir.mkdir(parents=True, exist_ok=True)
        self.book_cache_dir = CACHE_DIR / book_name
        self.book_cache_dir.mkdir(parents=True, exist_ok=True)

        self.storage = FileStorage(
            first_entry_file_name=str(self.book_cache_dir / "input.jsonl"),
            cache_path=str(self.book_cache_dir),
            file_name_prefix="vqa",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url=os.getenv("MINIMAX_API_URL", "https://api.minimaxi.com/v1/chat/completions"),
            model_name=os.getenv("MINIMAX_MODEL", "MiniMax-M1"),
            key_name_of_api_key="DF_API_KEY",
            max_workers=int(os.getenv("SHEZHEN_VQA_MAX_WORKERS", "5")),
            timeout=300,
        )

        self.pdf_merger = PDF_Merger(output_dir=str(self.book_cache_dir))
        self.mineru_out_dir = str(self.book_cache_dir / "mineru_output")
        self.mineru_backend = os.getenv("SHEZHEN_MINERU_BACKEND", "vlm-auto-engine")
        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChapterBasedGenerator(
            llm_serving=self.llm_serving,
            system_prompt=SHEZHEN_VQA_PROMPT,
            max_chunk_len=40000,
        )
        self.llm_output_parser = ZhaoLiMingOutputParser(
            output_dir=str(self.book_cache_dir),
            intermediate_dir="intermediate",
        )
        self.qa_merger = QA_Merger(
            output_dir=str(self.archive_output_dir),
            strict_title_match=False,
        )

    @staticmethod
    def _sanitize_llm_output_to_xml(raw_text: str) -> str:
        """Normalize model output so the downstream XML parser can consume it."""
        text = raw_text or ""
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

        # If provider wraps the final answer in <answer>, keep only those parts.
        answer_parts = re.findall(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
        if answer_parts:
            text = "\n".join(answer_parts)

        # Remove markdown code fences while keeping inner content.
        text = re.sub(r"```(?:xml)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
        return text.strip()

    def _normalize_llm_outputs(self, storage_step, input_key: str, output_key: str) -> None:
        """
        Read extracted LLM text paths, sanitize each file, and write a normalized path.
        This avoids parser failures caused by <think> wrappers and code fences.
        """
        df = storage_step.read("dataframe")
        normalized_count = 0
        for idx, row in df.iterrows():
            src = str(row.get(input_key, "")).strip()
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue
            raw = src_path.read_text(encoding="utf-8", errors="replace")
            cleaned = self._sanitize_llm_output_to_xml(raw)
            dst_path = src_path.with_name(src_path.stem + "_normalized.xml")
            dst_path.write_text(cleaned, encoding="utf-8")
            df.at[idx, output_key] = str(dst_path)
            normalized_count += 1
        print(f"[规范化] 清洗 LLM 输出 {normalized_count} 条")
        storage_step.write(df)

    @staticmethod
    def _infer_llm_output_path(converted_layout_path: str) -> Path:
        """
        ChapterBasedGenerator 默认把输入 json 路径替换为 *_llm_output.txt。
        这里按同一规则推断缓存路径。
        """
        return Path(str(converted_layout_path).replace(".json", "_llm_output.txt"))

    def _reuse_cached_llm_outputs(self, storage_step, converted_key: str, llm_key: str) -> bool:
        """
        若本书所有样本已有 LLM 输出文件，则直接回填路径并跳过 API 生成。
        返回 True 表示已复用，False 表示需要重新调用 LLM。
        """
        df = storage_step.read("dataframe")
        if df.empty:
            return False

        all_hit = True
        hit_count = 0
        for idx, row in df.iterrows():
            converted_path = str(row.get(converted_key, "")).strip()
            if not converted_path:
                all_hit = False
                continue
            cached_llm_path = self._infer_llm_output_path(converted_path)
            if cached_llm_path.exists():
                df.at[idx, llm_key] = str(cached_llm_path)
                hit_count += 1
            else:
                all_hit = False

        if all_hit:
            storage_step.write(df)
            print(f"[复用] 命中历史 LLM 输出 {hit_count} 条，跳过 API 生成")
            return True
        return False

    def _sample_layout_for_debug(self, storage_step, input_key: str, output_key: str) -> None:
        """
        For low-cost debugging: keep only the first N images and nearby context.
        If sample_images is None or <=0, pass through original layout path.
        """
        df = storage_step.read("dataframe")
        sample_n = self.sample_images or 0
        if sample_n <= 0:
            for idx, row in df.iterrows():
                df.at[idx, output_key] = row.get(input_key, "")
            storage_step.write(df)
            return

        for idx, row in df.iterrows():
            src = str(row.get(input_key, "")).strip()
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue

            items = json.loads(src_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(items, list):
                df.at[idx, output_key] = src
                continue

            image_indices = [i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") == "image"]
            picked = image_indices[:sample_n]
            if not picked:
                df.at[idx, output_key] = src
                continue

            selected_indices = set()
            for i, it in enumerate(items):
                if isinstance(it, dict) and it.get("type") == "header":
                    selected_indices.add(i)
            for img_idx in picked:
                left = max(0, img_idx - self.sample_window)
                right = min(len(items), img_idx + self.sample_window + 1)
                selected_indices.update(range(left, right))

            sampled = [items[i] for i in sorted(selected_indices)]
            # Keep sampled file in short cache path to avoid Windows long-path issues.
            dst_path = self.book_cache_dir / f"sampled_layout_row{idx}_{sample_n}img.json"
            dst_path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
            df.at[idx, output_key] = str(dst_path)
            print(f"[抽样] {src_path.name}: 原图{len(image_indices)}张 -> 抽样{len(picked)}张, layout={dst_path.name}")

        storage_step.write(df)

    def _print_llm_debug(self, storage_step, input_key: str) -> None:
        """
        Call LLM on sampled layout and print raw output to terminal.
        This mode avoids downstream parser/storage to save debugging cost and time.
        """
        df = storage_step.read("dataframe")
        if df.empty:
            print("[调试] 无可用输入")
            return

        all_inputs: list[str] = []
        for _, row in df.iterrows():
            path = str(row.get(input_key, "")).strip()
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                continue
            items = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(items, list):
                continue

            chapters = self.vqa_extractor._split_by_chapter(items)
            chunks = []
            for chapter in chapters:
                chunks.extend(self.vqa_extractor._split_recursive_items(chapter))

            for chunk_items in chunks[: max(1, self.debug_chunks)]:
                chunk_text = json.dumps(chunk_items, ensure_ascii=False, indent=2)
                all_inputs.append(SHEZHEN_VQA_PROMPT + "\n" + chunk_text)

        if not all_inputs:
            print("[调试] 未构建出可请求的 chunk")
            return

        print(f"[调试] 即将请求 LLM: {len(all_inputs)} 个 chunk")
        responses = self.llm_serving.generate_from_input(all_inputs)
        for i, resp in enumerate(responses, start=1):
            print("\n" + "=" * 30 + f" LLM CHUNK {i} " + "=" * 30)
            print(resp if resp else "[空响应]")
            print("=" * 75)

    def forward(self):
        self.pdf_merger.run(
            storage=self.storage.step(),
            input_pdf_list_key="input_pdf_paths",
            input_name_key="name",
            output_pdf_path_key="merged_pdf_path",
        )
        # 本地 MinerU 转换（断点续转）
        mineru_step = self.storage.step()
        df = mineru_step.read("dataframe")
        for idx, row in df.iterrows():
            md_path = run_local_mineru(
                row["merged_pdf_path"],
                self.mineru_out_dir,
                self.mineru_backend,
                book_name=self.book_name,
            )
            df.at[idx, "vqa_markdown_path"] = md_path
        mineru_step.write(df)
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="vqa_markdown_path",
            output_converted_layout_key="converted_vqa_layout_path",
        )
        # 低成本调试：只保留前 N 张图及其邻域上下文。
        self._sample_layout_for_debug(
            storage_step=self.storage.step(),
            input_key="converted_vqa_layout_path",
            output_key="effective_vqa_layout_path",
        )
        if self.print_only:
            self._print_llm_debug(
                storage_step=self.storage.step(),
                input_key="effective_vqa_layout_path",
            )
            print("[调试] print-only 模式结束，不写入解析结果")
            return
        llm_step = self.storage.step()
        reused = False
        if self.skip_llm_if_cached:
            reused = self._reuse_cached_llm_outputs(
                storage_step=llm_step,
                converted_key="effective_vqa_layout_path",
                llm_key="extracted_llm_vqa_path",
            )
        if not reused:
            self.vqa_extractor.run(
                storage=llm_step,
                input_path_key="effective_vqa_layout_path",
                output_path_key="extracted_llm_vqa_path",
            )
        else:
            print("[跳过] 使用缓存，不调用 LLM API")
        # 先规范化一遍模型输出，减少解析器与输出格式不匹配的问题。
        self._normalize_llm_outputs(
            storage_step=self.storage.step(),
            input_key="extracted_llm_vqa_path",
            output_key="normalized_llm_vqa_path",
        )
        self.llm_output_parser.run(
            storage=self.storage.step(),
            input_response_path_key="normalized_llm_vqa_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_vqa_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_qalist_path_key="extracted_vqa_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_vqalist_path",
            output_merged_md_path_key="output_merged_md_path",
            output_qa_item_key="vqa_pair",
        )


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="舌诊 VQA 生成")
    parser.add_argument("--book", type=str, default=None,
                        help="只处理指定书名（如 中医舌诊彩色图谱_龚一萍），默认处理全部")
    parser.add_argument("--force-llm", action="store_true",
                        help="忽略历史 LLM 输出缓存，强制重新调用 API")
    parser.add_argument("--sample-images", type=int, default=0,
                        help="调试模式：每本书仅抽取前 N 张图片参与 LLM 生成（0 表示不抽样）")
    parser.add_argument("--sample-window", type=int, default=20,
                        help="调试模式：每张抽样图片保留前后多少个 layout item 作为上下文")
    parser.add_argument("--print-only", action="store_true",
                        help="仅打印模型输出到终端，不进行解析与结果落盘")
    parser.add_argument("--debug-chunks", type=int, default=1,
                        help="print-only 模式下最多请求多少个 chunk")
    parser.add_argument("--archive-subdir", type=str, default="",
                        help="归档输出到 output 下的新子目录，例如 rerun_20260402")
    args = parser.parse_args()

    api_key = os.getenv("DF_API_KEY") or os.getenv("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("请设置环境变量 DF_API_KEY 或 MINIMAX_API_KEY")
    os.environ["DF_API_KEY"] = api_key

    archive_output_dir = OUTPUT_DIR
    if args.archive_subdir.strip():
        safe_subdir = re.sub(r"[^\w\-.]+", "_", args.archive_subdir.strip()).strip("._")
        if not safe_subdir:
            raise ValueError("--archive-subdir 经过清洗后为空，请换一个名字")
        archive_output_dir = OUTPUT_DIR / safe_subdir
        archive_output_dir.mkdir(parents=True, exist_ok=True)

    # 过滤书籍列表
    books_to_run = TARGET_BOOKS
    if args.book:
        books_to_run = [b for b in TARGET_BOOKS if b["name"] == args.book]
        if not books_to_run:
            print(f"未找到书籍: {args.book}")
            print(f"可用书籍: {[b['name'] for b in TARGET_BOOKS]}")
            return

    print(f"舌诊书籍目录: {SHEZHEN_BOOKS_DIR}")
    print(f"工作目录    : {WORK_DIR}")
    print(f"处理 {len(books_to_run)} 本书籍")

    results = {}
    for book in books_to_run:
        book_name = book["name"]
        print(f"\n{'='*60}")
        print(f"[开始] {book_name}")

        # 检查 PDF 是否存在
        pdf_path = SHEZHEN_BOOKS_DIR / book["pdf"]
        if not pdf_path.exists():
            print(f"[跳过] PDF 不存在: {pdf_path.name}")
            results[book_name] = "跳过（PDF不存在）"
            continue

        # 检查是否已完成（空文件不视为完成，允许重跑）
        output_file = archive_output_dir / book_name / "vqa_dataset.jsonl"
        if output_file.exists() and (not args.print_only):
            count = sum(1 for line in output_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            if count > 0:
                print(f"[跳过] 已完成，共 {count} 条 VQA")
                results[book_name] = f"已完成（{count} 条）"
                continue
            print("[重跑] 检测到历史输出为空，重新生成")

        # 写 input.jsonl
        book_cache = CACHE_DIR / book_name
        book_cache.mkdir(parents=True, exist_ok=True)
        input_file = book_cache / "input.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(
                {"input_pdf_paths": str(pdf_path), "name": book_name},
                ensure_ascii=False
            ) + "\n")

        try:
            pipeline = ShezhenVQAPipeline(
                book_name=book_name,
                skip_llm_if_cached=(not args.force_llm),
                sample_images=(args.sample_images if args.sample_images > 0 else None),
                sample_window=args.sample_window,
                print_only=args.print_only,
                debug_chunks=max(1, args.debug_chunks),
                archive_output_dir=archive_output_dir,
            )
            # This pipeline contains imperative storage reads/writes inside `forward`.
            # Calling `compile()` would execute a dry-run graph build and trigger false missing-step errors.
            pipeline.forward()
            results[book_name] = "成功"
        except Exception as e:
            print(f"[异常] {book_name}: {e}")
            results[book_name] = f"异常: {e}"

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VQA 提取汇总报告")
    print("="*60)
    for book, status in results.items():
        icon = "v" if "成功" in status or "已完成" in status else "x"
        print(f"  [{icon}] {book}: {status}")
    print("="*60)


if __name__ == "__main__":
    main()
