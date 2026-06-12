from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（《舌诊辨证图谱第2版》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 第一章：舌诊辨证基础知识
  - 第一节 舌的形态、结构
  - 第二节 舌诊原理
  - 第三节 舌诊的意义
  - 第四节 舌诊方法
- 第二章：舌象辨证诊病
  - 第一节 舌质辨证诊病
  - 第二节 舌苔辨证诊病
  - 第三节 舌脉辨证诊病
  - 第四节 舌纹辨证诊病
  - 第五节 舌觉辨证诊病
- 第三章：舌诊临床辨证各论
  - 含多系统疾病分节（传染/呼吸/消化/心脑血管/血液与结缔组织/内分泌代谢/神经/泌尿/妇科/男科/运动系统等）
  - 常见模式为“疾病名 + 舌象图 + 辨证意义/病机/诊断提示”
- 本书可能出现目录页、页码干扰、跨页续文；图文需按就近和同小节配对
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
2. Omit incomplete units that continue on the next page. If only the long remedy text is incomplete, you may keep 辨证/舌象 and short 治法, solution empty.
3. Treat unmarked opening paragraphs as continuation from the previous page unless they clearly start a new subsection.
4. Include all section titles visible on page_n.
"""
        PROMPT += f"""
When the page has two columns, read **left to right**, then top to bottom; output in the same order.
Document hierarchy hint (《舌诊辨证图谱第2版》; do not invent text): Chapter 1 covers tongue-diagnosis basics (shape/structure, principles, meaning, methods). Chapter 2 covers syndrome differentiation by tongue signs (舌质/舌苔/舌脉/舌纹/舌觉). Chapter 3 covers disease-specific clinical differentiation across systems. Pair image and text by local section context and nearest explanatory lines.
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
    """MinerU content_list JSON → LLM：以图片为中心，每张有效舌诊教学图须生成 3～5 个 QA（硬性目标）。"""

    def __init__(self):
        pass

    def build_prompt(self) -> str:
        PROMPT = f"""
你是《舌诊辨证图谱第2版》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。问答须紧扣中文原文与图注，输出全部为中文。

**零容忍输出约束（优先级最高）**
- 只允许输出最终 XML 结果，不得输出任何思考/草稿/解释文本。
- 若输出了任何非 XML 正文，将导致解析失败。请直接从 `<chapter>` 或 `<empty></empty>` 开始。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中每个 `img_path`，再围绕该图在顺序上附近找文字（图下标注、上下行说明、同小节标题）生成 QA。
- **文字为辅**：不要先写大量无图 QA；有图时每条 QA 都必须绑定对应 `<pic>`。
- **`id` 递增≈阅读顺序**：就近配对图文，避免跨页跨段借证据。
- **语言约束（强制）**：`<title>/<question>/<answer>/<solution>/<source_text>` 全部中文。

{_TONGUE_BOOK_OUTLINE}

**核心任务**
- **第一章整体跳过**：凡属于第一章（舌诊辨证基础知识）或页码 < 41 的内容，一律不生成 QA。
- 从**第二章（第41页起）**开始生成，并严格按章节类别限制 QA 类型：
  1) **第二章（舌象辨证诊病）只允许 3 类 QA**  
     - 舌象特征类  
     - 提示的病症/证候类  
     - 形成原因/病机类  
  2) **第三章（舌诊临床辨证各论）只允许 2 类 QA**  
     - 舌象特征类  
     - 提示的病症/证候类  
- 除上述类型外，**禁止生成任何其他类型 QA**（如治法、预后、护理、鉴别、方药等）。
- 某一类型若证据不足可不生成；有证据才生成，不得臆造。
- 章节判定优先：标题文本 > 就近小节 > 页码线索。
- 每一类 QA 参考以下**典型示例问句**。生成时可改写措辞，但语义必须与对应类别一致：
  - **第二章-舌象特征类（示例）**
    - `图中舌象有哪些可见特征？`
    - `该图舌质与舌苔表现如何概括？`
    - `该图可见哪些舌象形态特征？`
  - **第二章-提示病症/证候类（示例）**
    - `该图舌象提示什么病症？`
    - `该图舌象多见于哪类证候？`
    - `该图舌象反映了什么病证？`
  - **第二章-形成原因/病机类（示例）**
    - `该图舌象形成的原因是什么？`
    - `该图舌象提示什么病机？`
    - `该图所示舌象多由何种病理所致？`
  - **第三章-舌象特征类（示例）**
    - `图中舌象特征如何概括？`
    - `该图可见哪些舌质舌苔特征？`
    - `该图对应的舌象表现是什么？`
  - **第三章-提示病症/证候类（示例）**
    - `该图舌象提示什么病症？`
    - `该舌象与哪类疾病/证候相关？`
    - `该图舌象反映了什么病证？`

**图片处理规则（最重要）**
1. 每个有效 `img_path` 生成 3～5 条 `<qa_pair>`（证据不足时可少于 3，但不得只写明显凑数的空话）。
2. 每条 `<qa_pair>` 内恰好一个 `<pic>`，且 `<pic>` 必须与该条问题语义主体一致。
3. 图文配对优先级（本书定制）：
   - **第二章**：以图号所在句为中心，默认取“前2句 + 后5句”作为证据窗；可根据内容完整性微调（例如前1后6、前3后4）。
   - **第三章**：以图号所在句为主；必要时只参考前后各1句，不得大范围跨段取证。
   - 若窗口内仍无可靠证据，跳过该图该类型 QA。
4. 单图单问：禁止把多张图合并在一条问题中。
5. 非舌诊教学图（纯装饰图、与舌诊无关的人体/器械图）跳过不生成；若图与疾病条目相关且有明确舌诊文字证据，可保留。

**舌象措辞与一致性（强制）**
- 问句中的舌象措辞优先取该图图注与紧邻说明；冲突时以图注为准。
- 不得把邻近其它条目的舌象词错贴到当前图。
- 若无法确定具体细节，问句仅用「图中舌象」等中性表达，不得编造苔色/津液等细节。
- 问句优先避免出现图号；若原文仅用图号定位且不写图名，可用“该图/图中舌象”表达，不抄图号。

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
4. 禁止生成与舌诊无关图片的 QA。
5. 第二章仅允许“舌象特征 / 病症提示 / 形成原因（病机）”三类。
6. 第三章仅允许“舌象特征 / 病症提示”两类；不得输出病机类。
7. 第一章与第41页前内容一律不生成。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；一图可多条 `<qa_pair>`；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>第二章 舌象辨证诊病——某舌象条目</title>
<qa_pair><label>1</label><question>图中舌象有哪些可见特征？</question><answer>舌质偏红，苔薄黄。</answer><solution>依据图号附近前后文，舌质偏红、苔薄黄。</solution><source_text>id:xxx 图号句；id:yyy 前后句原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>该图舌象提示什么病症？</question><answer>提示热证倾向。</answer><solution>依据同段原文，舌象与热证相关。</solution><source_text>id:yyy 对应原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>3</label><question>该图舌象形成的原因是什么？</question><answer>多由热邪内盛、津液受损所致。</answer><solution>第二章允许病机类问题，证据取图号句前后窗口。</solution><source_text>id:zzz 病机原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图A和图B分别提示什么？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图A 一条、图B 一条、图C 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<redacted_thinking>`/analysis/draft text.
Reminder: skip Chapter 1 entirely; start from Chapter 2 (page 41). Enforce QA type whitelist by chapter exactly.
"""
        return PROMPT

