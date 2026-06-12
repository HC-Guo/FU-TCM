from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（《实用中医舌诊彩色图谱》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 舌诊基本知识（前48页，含：舌的结构与功能、舌与脏腑经络关系、望舌原理/意义/方法/内容）【本任务整体跳过】
- 典型舌象图谱（从第49页开始）：
  - 正常舌象
  - 望舌质（颜色、形态、动态等）
  - 望舌苔（苔色、苔质）
  - 舌质和舌苔综合诊察
  - 以及舌下络脉等相关条目
- 本书大量图片无明确图号，图文对应主要依赖“同页就近文本”
- 可能出现目录页、页眉页脚、跨页续文；图文必须按同页配对，禁止跨页借证据
"""


@PROMPT_REGISTRY.register()
class VQAExtractPrompt(PromptABC):
    """按页图像 + 检测框标注抽取时使用（与 MinerU-JSON 流水线不同）。"""

    def __init__(self):
        pass

    def build_prompt(
        self,
        example_title,
        subject: str = "Traditional Chinese Medicine (TCM) tongue diagnosis",
        interleaved=True,
    ) -> str:
        PROMPT = ""
        if interleaved:
            PROMPT = f"""
        You are an expert in {subject} and clinical tongue-image interpretation. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All teachable units relevant to tongue diagnosis, TCM pattern differentiation (辨证), treatment principles, and simple treatments (方药/针灸/外治等) whose text **begins** on page_n, together with explanations that belong to those units.
2. If a unit or its explanation is cut off and continues on page_n+1, omit the incomplete unit. If the clinical description is complete but the treatment paragraph is not, you may keep the description and leave solution empty; if the core unit is incomplete, omit the whole unit.
3. A block at the top of the page without a clear section marker (续上页、接上文、无标题段落) is usually continuation from the previous page—omit unless it clearly starts a new numbered subsection on this page.
4. Section/chapter headings as they appear on page_n. Include headings even when the page only has titles or introductory lines under them.
"""
        else:
            PROMPT = f"""
        You are an expert in {subject}. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Extract from page_n only:
1. All teachable units whose text appears on page_n (the page may be mostly questions, mostly answers, or interleaved—follow what is actually on the page).
2. Omit incomplete units that continue on the next page. If only long remedy text is incomplete, you may keep concise diagnostic description and short treatment principle, leaving solution empty.
3. Treat unmarked opening paragraphs as continuation from the previous page unless they clearly start a new subsection.
4. Include all section titles visible on page_n.
"""
        PROMPT += f"""
When the page has two columns, read **left to right**, then top to bottom; output in the same order.
Document hierarchy hint (《实用中医舌诊彩色图谱》; do not invent text): foundational tongue-diagnosis knowledge in early pages should be skipped for QA generation; from the typical tongue-atlas section onward, pair each tongue image with same-page nearby explanatory lines and avoid cross-page matching.
Strict extraction rules:
** Units (questions / answers / solutions) **
- If the page is cover, copyright, pure catalog, page numbers only, or irrelevant front/back matter, output `<empty></empty>`.
- **Labels**: preserve the book’s markers, e.g. "（一）", "（二）", "一、", "1.", "例1". Prefer Arabic digits only when the book mixes styles (例一→例1). For headings like "（三）钩虫病", use that as label for the following qa_pair(s).
- Multiple sub-points (1)(2) or (1)(a) under one disease block: keep them in **one** `<qa_pair>`…`</qa_pair>`.
- If 舌象、辨证、治法、方药 are contiguous, one `<qa_pair>` with question/solution split sensibly; if clearly separated by headings, use separate `<qa_pair>` blocks with appropriate labels.
- Use `<answer>` for very short outcomes (e.g. 治则一词) when the book separates them; use `<solution>` for longer 方药、煎服法、注意事项.
** Chapter / section titles (text in <title>, this prompt mode) **
- `<chapter>`…`</chapter>` with `<title>…</title>` = the section heading text on the page (e.g. "注意与染苔的鉴别" or "2.7 黄腻苔" or "附录一 斑疹白瘖"). Multiple chapters if multiple such headings appear.
- For title style, follow the example: "{example_title}" when a single canonical form is needed.
- If a title appears at page end with no body, still output the chapter with `<qa_pair><label>0</label><question></question><answer></answer><solution></solution></qa_pair>`.
- Do not nest titles; no duplicate hierarchy in one title string.
** Text and figures **
- This book is Chinese-English parallel text. For downstream QA, keep and output **Chinese only**; ignore side-by-side English translation unless a term has only English and no Chinese counterpart.
- Use LaTeX only for real formulas; plain text for pinyin or terms.
- For figures/tongue photos, use `<pic>tagA:boxB</pic>` exactly as labeled on the image (red box tags). Place tags where the text refers to the figure. Never invent tags.

If nothing qualifies: `<empty></empty>`

Output format (tags contiguous, minimal extra newlines):
<chapter><title>MAIN_TITLE</title>
<qa_pair><label>…</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>
</chapter>

Example:
<chapter><title>1.2 舌与脏腑经络的关系</title>
<qa_pair><label>1</label><question>据图中舌象，本书如何概括该舌象或结构要点？<pic>tag2:box5</pic></question><answer>按该图图注给出结论（中文）。</answer><solution>结合邻文补充辨析或临床意义（中文）。</solution></qa_pair>
</chapter>

Please now process the provided page_n image and output your result.
"""
        return PROMPT


@PROMPT_REGISTRY.register()
class QAExtractPrompt(PromptABC):
    """MinerU content_list JSON → LLM：以图片为中心，按部位证据生成高质量 QA。"""

    def __init__(self):
        pass

    def build_prompt(self) -> str:
        PROMPT = f"""
你是《实用中医舌诊彩色图谱》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。问答须紧扣中文原文与图注，输出全部为中文。

**零容忍输出约束（优先级最高）**
- 只允许输出最终 XML 结果，不得输出任何思考/草稿/解释文本。
- 若输出了任何非 XML 正文，将导致解析失败。请直接从 `<chapter>` 或 `<empty></empty>` 开始。
- 严禁输出以下任一标签或包裹：`<think>`、`<thinking>`、`<redacted_thinking>`、`<answer>`（外层包裹）、Markdown 代码块（```）。
- 严禁“先 `<empty></empty>` 再继续输出 `<chapter>`”；二者只能二选一。
- 输出后不得追加任何尾注/解释/第二版本结果；只保留一版最终 XML。
- `<empty></empty>` 仅允许在“确无可抽取内容”时使用；若存在可用图片与证据，禁止以 `<empty></empty>` 代替。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中每个 `img_path`，再围绕该图在顺序上附近找文字（图下标注、上下行说明、同小节标题）生成 QA。
- **文字为辅**：不要先写大量无图 QA；有图时每条 QA 都必须绑定对应 `<pic>`。
- **`id` 递增≈阅读顺序**：就近配对图文，避免跨页跨段借证据。
- **语言约束（强制）**：`<title>/<question>/<answer>/<solution>/<source_text>` 全部中文。

{_TONGUE_BOOK_OUTLINE}

**核心任务**
- **前48页整体跳过**：凡属于“舌诊基本知识”部分（前48页）的内容，一律不生成 QA。
- 从**第49页（典型舌象图谱）开始**生成。
- 全书只允许 2 类 QA：
  - **舌象特征类**（该图舌象有什么特征/表现）
  - **主病提示类**（该图舌象提示什么病症/主病）
- 除上述两类外，**禁止生成任何其他类型 QA**（如病机、治法、预后、护理、方药、鉴别、操作步骤等）。
- 某一类型若证据不足可不生成；有证据才生成，不得臆造。
- 章节判定优先：标题文本 > 就近小节 > 页码线索。
- 每一类 QA 参考以下**典型示例问句**。生成时可改写措辞，但语义必须与对应类别一致：
  - **舌象特征类（示例）**
    - `图中舌象有哪些可见特征？`
    - `该图舌质与舌苔表现如何概括？`
    - `该图舌体形态与颜色有什么特征？`
  - **主病提示类（示例）**
    - `该图舌象提示什么主病？`
    - `该图舌象反映了什么病证？`

**舌象精准提问（必须遵守）**
- 问句必须围绕当前图的舌象关键词（如淡白、红绛、胖嫩、裂纹、黄腻苔、少苔等）组织，不得泛化为空话。
- 不要在问题中直接给出诊断结论，避免覆盖答案；问题“具体但不泄露结论”。
- 从图注和邻近原文提取关键词做同义改写，保持语义不变，不得扩展原文没有的信息。

**图片处理规则（最重要）**
1. 每个有效 `img_path` 优先生成 2 条 `<qa_pair>`（舌象特征类 1 条 + 主病提示类 1 条）；若某一类证据不足可只生成另一类，不得臆造。
2. 每条 `<qa_pair>` 内恰好一个 `<pic>`，且 `<pic>` 必须与该条问题语义主体一致。
3. 图文配对优先级（本书定制）：
   - 本书很多图片没有图号，**以页面为最小配对单元**：只允许使用与该 `img_path` 同一 `page_idx` 的文本作为证据。
   - 同页内按 `id` 就近原则取证：优先图下说明、同段文字、前后邻近句（建议前2句+后3句）。
   - **严禁跨页匹配**：不得引用其他 `page_idx` 的文本解释当前图片。
   - 若同页内仍无可靠证据，跳过该图该类型 QA。
4. 单图单问：禁止把多张图合并在一条问题中。
5. 非教学相关图（纯装饰图、与望诊无关的人体/器械图）跳过不生成；若图与疾病条目相关且有明确望诊文字证据，可保留。

**`<empty></empty>` 触发条件（严格）**
-- 只有当以下条件**同时满足**时，才允许输出 `<empty></empty>`：
  1) 本段 JSON 中不存在可用 `img_path`；
  2) 不存在可抽取的中文证据文本（同页图注/同页邻近说明）；
  3) 内容属于前48页基础知识或其他明显无关内容（目录、版权、页眉页脚等）。
- 只要本段存在任一可用图片且可找到证据，必须输出 `<chapter>...</chapter>`，禁止输出 `<empty></empty>`。
- 若存在图片但证据较弱，不得直接判空；应输出“最小可用结果”：至少 1 个 `<chapter>` + 1 条带 `<pic>` 与 `<source_text>` 的 `<qa_pair>`（仅用中性、不臆造的表述）。

**舌象措辞与一致性（强制）**
- 问句中的舌象措辞优先取该图图注与同页紧邻说明；冲突时以图注为准。
- 不得把同页其它图片或跨页条目的舌象词错贴到当前图。
- 若无法确定具体细节，问句仅用「图中舌象表现」等中性表达，不得编造苔色/津液等细节。
- 问句优先避免出现图号；若原文无图号，直接用“该图舌象/图中舌象”表达。

**问答生成（字段分工）**
- `<question>`：只围绕当前单图；使用无图号表达；避免“见图/请参考该图”等赘述。
- `<answer>`：短答，先给结论，再尽量补一句依据（来自同图附近原文）。
- `<solution>`：长解释，摘录该图所在小节的原文释义/意义要点；无证据可留空，不得编造。
- `<source_text>`：必须逐字摘录输入 JSON 的可见原文，标注 `id:数字`，不得用“同上/略”占位。
- `<label>`：本段输出内从 1 连续递增。

**绝对约束（下游解析）**
1. 只输出 `<chapter>...</chapter>` 或 `<empty></empty>`。
2. 有图时，每条 `<qa_pair>` 都必须有且仅有一个 `<pic>`，并包含 `<source_text>`。
3. 禁止多图合问、禁止无依据外推、禁止输出与最终 XML 无关的前后缀。
4. 禁止生成与舌诊主题无关图片的 QA。
5. 仅允许“舌象特征 / 主病提示”两类。
6. 禁止输出病机类、治法类、方药类等任何其他类别。
7. 前48页内容一律不生成；第49页起按可见证据生成。
8. 所有标签必须闭合且顺序合法：`<chapter><title>...</title><qa_pair>...</qa_pair>...</chapter>`；禁止半截标签和未闭合标签。
9. 不得复制“示例文本”或“任务说明”到结果中；仅输出从输入 JSON 证据生成的内容。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；一图可多条 `<qa_pair>`；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>典型舌象图谱——某舌象条目</title>
<qa_pair><label>1</label><question>图中舌象有哪些可见特征？</question><answer>可见舌质淡白，舌体偏胖嫩，苔白腻。</answer><solution>依据同页图注与邻近说明，该图主要表现为舌质淡白、舌体胖嫩及白腻苔。</solution><source_text>id:xxx 同页图注；id:yyy 同页邻近句原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>该图舌象提示什么主病？</question><answer>提示脾阳不足、痰湿内停相关病证。</answer><solution>依据同页原文，该舌象与脾虚湿盛或痰湿内阻等主病线索相关。</solution><source_text>id:yyy 同页对应原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图A和图B分别提示什么？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图A 一条、图B 一条、图C 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`.
Reminder: never output `<think>`, `<thinking>`, `<redacted_thinking>`, outer `<answer>`, or markdown code fences.
Reminder: choose exactly one mode: either valid `<chapter>...</chapter>` blocks OR a single `<empty></empty>`; never both.
Reminder: `<empty></empty>` is only allowed when all strict empty conditions are met; if uncertain, output a minimal valid chapter instead of empty.
Reminder: skip the first 48 pages entirely; generate from page 49 (典型舌象图谱) only.
Reminder: allow only two QA categories globally: 舌象特征 and 主病提示.
Reminder: do page-level matching only; never use cross-page evidence for any image.
"""
        return PROMPT

