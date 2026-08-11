from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（《温病舌诊图谱 第2版》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 温病舌诊的注意事项
  - 注意舌象的动态变化（苔色、厚薄、润燥、舌色、舌质荣枯）
  - 注意季节变化、昼夜时辰对舌象变化的影响
  - 注意与染苔的鉴别
- 正常舌象
- 温病常见舌象变化
  - 白苔（薄白欠润、薄白而干、白厚黏腻、白砂苔等）
  - 黄苔（薄黄不燥、黄白相兼、黄燥苔、老黄苔、黄腻苔等）
  - 灰苔（灰燥、灰腻、灰滑）
  - 黑苔（焦燥起刺、干燥、如烟煤隐隐等）
  - 红舌、绛舌、紫舌、舌疮、淡白舌、染苔
- 附录
  - 斑、疹及相关分型
  - 吴坤安察舌辨症歌
  - 方剂索引、斑疹白喉
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
Document hierarchy hint (《温病舌诊图谱 第2版》; do not invent text): 注意事项、正常舌象、温病常见舌象变化（白苔/黄苔/灰苔/黑苔/红舌/绛舌/紫舌/舌疮/淡白舌/染苔）与附录。Prefer section titles as printed on the page.
Strict extraction rules:
** Units (questions / answers / solutions) **
- If the page is cover, copyright, pure catalog, page numbers only, or irrelevant front/back matter, output `<empty></empty>`.
- **Labels**: preserve the book’s markers, e.g. "（一）", "（二）", "一、", "1.", "例1". Prefer Arabic digits only when the book mixes styles (例一→例1). For headings like "（三）钩虫病", use that as label for the following qa_pair(s).
- Multiple sub-points (1)(2) or (1)(a) under one disease block: keep them in **one** `<qa_pair>`…`</qa_pair>`.
- If 舌象、辨证、治法、方药 are contiguous, one `<qa_pair>` with question/solution split sensibly; if clearly separated by headings, use separate `<qa_pair>` blocks with appropriate labels.
- Use `<answer>` for very short outcomes (e.g. 治则一词) when the book separates them; use `<solution>` for longer 方药、煎服法、注意事项.
** Chapter / section titles (text in <title>, this prompt mode) **
- `<chapter>`…`</chapter>` with `<title>…</title>` = the section heading text on the page (e.g. "二、舌色" or "慢性胃炎—脾胃气虚证" or "一、气虚质"). Multiple chapters if multiple such headings appear.
- For title style, follow the example: "{example_title}" when a single canonical form is needed.
- If a title appears at page end with no body, still output the chapter with `<qa_pair><label>0</label><question></question><answer></answer><solution></solution></qa_pair>`.
- Do not nest titles; no duplicate hierarchy in one title string.
** Text and figures **
- Keep original Chinese (and English if printed). Use LaTeX only for real formulas; plain text for pinyin or terms.
- For figures/tongue photos, use `<pic>tagA:boxB</pic>` exactly as labeled on the image (red box tags). Place tags where the text refers to the figure. Never invent tags.

If nothing qualifies: `<empty></empty>`

Output format (tags contiguous, minimal extra newlines):
<chapter><title>MAIN_TITLE</title>
<qa_pair><label>…</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>
</chapter>

Example:
<chapter><title>慢性胃炎—脾胃气虚证</title>
<qa_pair><label>1</label><question>该证舌象特征与辨证要点。<pic>tag2:box5</pic></question><answer>舌淡胖有齿痕等（据原文）。</answer><solution>治法方药与调护（据原文）。</solution></qa_pair>
</chapter>

Please now process the provided page_n image and output your result.
"""
        return PROMPT


@PROMPT_REGISTRY.register()
class QAExtractPrompt(PromptABC):
    """MinerU content_list JSON → LLM：以图片为中心，每张有意义的图片至少生成一个 QA 对。"""

    def __init__(self):
        pass

    def build_prompt(self) -> str:
        PROMPT = f"""
你是《温病舌诊图谱 第2版》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。

**零容忍输出约束（优先级最高）**
- 严禁输出任何思考/草稿/解释文本，包括但不限于：`<think>`、`<thinking>`、`<reasoning>`、`analysis`、`Now, ...`、`Let's ...`、"我先..." 等元叙述。
- **模型在输出中写 `<think>` 或任何非 XML 内容属于严重违规**，将导致下游解析失败。
- 一旦输出任何上述内容，请立即停止。请直接从 `<chapter>` 或 `<empty></empty>` 开始输出，并在最后一个 `</chapter>` 后立即结束，不要添加任何额外内容。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中每个 `img_path`，再围绕该图在顺序上**附近**找文字（图注、上下行说明、同小节标题），生成 QA。
- **文字为辅**：不要为了凑条数先写无图 QA；本段内每张有效教学图都要产出基于该图的 `<qa_pair>`。
- **`id` 递增 ≈ 阅读顺序**，就近配对图注与正文。

{_TONGUE_BOOK_OUTLINE}

**核心任务：以图片为中心，为每张有意义的图片生成 1～3 个独立 `<qa_pair>`。每图优先覆盖三类题目：**
1) 图示舌头**特征**是什么；
2) 该特征的**意义/提示**是什么；
3) 若原文给出，**怎么治疗**（治法/方药/调护）是什么。
**依据输入决定题型组合**：
- 输入中有特征 → 出特征题
- 输入中有意义/提示 → 出意义题  
- 输入中有治疗/方药 → 出治疗题
三类题目各自独立，有证据就出，无证据不出；不强求每题都三题齐全，但证据充足时必须出全。

**图片处理规则（最重要）**
1. **每张图片至少一条（强制）**：输入 JSON 中每个有 `img_path` 且非装饰/页眉类的图片项，至少生成 1 条 `<qa_pair>`；证据充分时建议 2～3 条，优先覆盖“特征/意义/治疗”。
2. **一图多题拆分**：同一图可将“特征、意义、治疗”拆入不同 `<qa_pair>`；禁止把多张不同 `img_path` 合并进同一 `<qa_pair>`。
3. **图文配对优先级**（从高到低）：
   - 图注行（如「图1-1 舌背各部名称」「图2-1 …」）
   - 图片前最近的章节/小节标题
   - 图片后 1～3 行说明文字
   - 同章节内与图主题一致的提示句、证候句、治法句
4. **`<pic>` 输出（强制）**：
   - `<pic>` 必须使用输入 JSON 的 `img_path` 原文；
   - 一个 `<qa_pair>` 内 `<pic>` 只能出现一次。
4.1 **`<figure_id>` 输出（强制）**：
   - 每条 `<qa_pair>` 必须包含 `<figure_id>`，内容为该图在原文中的图号（如 `1.1.1`、`1.5.4`）。
   - `<figure_id>` 不要带“图”字，不要括号，不要其它文字，只保留数字与点号。
   - 若 `source_text` 同时出现多个图号且无法唯一对应当前 `<pic>`，则该条 `<figure_id></figure_id>` 留空，禁止猜测。
5. **问句不写图号（强制）**：`<question>` 禁止出现“图1-1/图2-3/（图x-x）”等图号字符串；只允许使用“图中/图中的舌象”这类表达。
6. **问句单图约束（强制）**：`<question>` 只能询问当前这一张图，禁止多图并提、对比两图/三图。
7. **证据驱动出题（强制）**：输入 JSON 中有什么证据，就出什么题目。例如：只有特征 → 只出特征题；有特征+意义 → 出两道题；有特征+意义+治疗 → 出三道题。允许一题、两题或三题，不强求数量，但证据充分时建议出全，避免遗漏。

**题目类型（硬约束）**
- 仅允许三类提问：特征类、意义类、治疗类。
- 推荐固定问法（可微调但语义不变）：
  - 特征类：「图中的舌象有什么特征？」
  - 意义类：「图中的舌象提示了什么意义？」
  - 治疗类：「针对图中的舌象应如何治疗？」
- 不允许“鉴别诊断有哪些/分几型/比较多图”等白名单外问法。
- 每条 `<question>` 只允许一个核心焦点（特征/意义/治疗三选一），禁止同题并列多个目标。
- 若问句含“和/及/与/分别/同时”等并列词且对应多个焦点，必须拆成多条 `<qa_pair>`。

**图文一致性与证据边界（强制）**
- 以该图对应的图注、邻近说明、同节原文为唯一证据来源。
- `<question>/<answer>/<solution>/<source_text>` 不得引入输入 JSON 中不存在的信息。
- 出现证据冲突时，以同图图注与同号锚点句为准。
- 治疗类答案必须有原文依据；无依据则不生成治疗题或将 `<solution>` 留空。
- 同段出现多个图号时，当前 `<pic>` 只能使用与本图同号的证据句；禁止跨图借证。

**问答生成（字段分工）**
- `<question>`：仅使用三类题目语义（特征/意义/治疗），且必须出现“图中/图中的”。
- `<answer>`：短答，只写与该题型直接相关的结论，不扩展到无依据内容。
- `<solution>`：用于补充原文中可追溯的意义解释或治疗细节；若原文无治疗，治疗题不出或 solution 为空。
- **`<source_text>`（强制，供人工/程序校验）**：**每条** `<qa_pair>` 末尾（建议放在 `</solution>` 与 `<pic>` 之间）必须输出 `<source_text>…</source_text>`。内容须为从**输入本段 JSON** 各条目的 **`text` 字段、`list_items`、`image_caption` 等中逐字复制**的可见正文（允许为连贯阅读做**换行/分号**连接，**不得改写证型/方名用字**）。**写法规范**：按条标注 `id:数字` 后**必须紧跟该行在 JSON 中的原文整句或连续片段**（每条 id 后正文一般不少于 15 字，原文更短时全文照录）。**严禁**仅用占位语代替正文，例如单独写「图注全文」「同上」「见 id:xxx（略）」「邻接 text 一句」而不贴出汉字原文。**禁止**在 `<source_text>` 内编造书中没有的句子；若无邻文可引，则至少**完整照录**该图 `image_caption` / 图注字符串本身（逐字）。
- **`<source_text>` 占位词零容忍（强制）**：禁止出现「图注原文」「第N条原文」「原文」「见原文」「同上原文」等占位写法；必须贴出可核验原句。
- **`<solution>` 不越权（强制）**：`<solution>` 不得新增未被题目覆盖的核心事实（尤其病机/主病提示）；若写入该类事实，必须补充对应“意义类”题目或删除该事实。
- `<figure_id>`：优先从同条 `source_text` 的同号锚点中提取；若仅有一个图号则填该号，若多个且无法唯一映射则留空。
- `<label>`：**本段输出内**从 1 起**全局连续递增**（一图多题时条数多，勿与上一输出段混号）；勿重复使用同一 `<label>` 绑定不同 `<question>`。

**绝对约束（下游解析）**
1. 只输出 `<chapter>`…`</chapter>` 或 `<empty></empty>`；禁止 JSON、禁止任何思考标签（如 `<think>`、`<thinking>`、`<reasoning>`、`<redacted_thinking>`）。
2. `<title>`、`<question>`、`<answer>`、`<solution>`、`<source_text>` 为自然语言字段；每条 `<qa_pair>` 必须包含 `<figure_id>`（可为空字符串）；`<source_text>` 内可带「id:数字」便于对照 JSON，但不要整段只输出 id 列表而无正文。
3. 禁止汇总型问句：如「该疾病对应的舌诊提示有哪些？」「本病舌诊要点有哪些？」。
4. **一图至少一题 + `<pic>` + `<figure_id>` + `<source_text>`（强制）**：本段 JSON 中每个有效 `img_path` 至少出现 1 次；每条 `<qa_pair>` 内必须且仅有一个 `<pic>`、一个 `<figure_id>`、一个 `<source_text>`。禁止多图合并同题。
5. 禁止输出任何与最终 XML 无关的前后缀文字（如“下面是结果”“说明如下”“继续”等）；输出首字符必须是 `<`，且应为 `<chapter>` 或 `<empty></empty>`。
6. 问句文本不得出现图号字符串（如“图1-1”“图2-7”）。
7. 问句中禁止出现未替换占位符（如 `[`、`]`）。

**禁止**
- 禁止把多张图合并进同一个 `<qa_pair>`。
- 禁止跳过本段 JSON 中应保留的教学图（仅装饰/页眉/纯重复可略）。
- 禁止编造不存在的 `img_path`。
- 禁止输出书中没有的结论、病机、治法、方药或调护建议。
- 禁止治疗题在无证据时硬答。
- 禁止在单条问题中混问“特征+意义+治疗”。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文的目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；每条含 `<figure_id>` 与 `<source_text>`；问题不写图号）：

<chapter><title>白苔</title>
<qa_pair><label>1</label><question>图中的舌象有什么特征？</question><answer>舌苔薄白欠润，舌边尖略红。</answer><solution></solution><figure_id>1.1.1</figure_id><source_text>id:142 图1.1.1 薄白欠润，舌边尖红。</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>图中的舌象提示了什么意义？</question><answer>提示温病初起或津液轻度受损。</answer><solution>据本节原文，该象提示邪在卫分并见轻度津伤。</solution><figure_id>1.1.1</figure_id><source_text>id:142 图1.1.1 薄白欠润，舌边尖红。id:143 本象多见于温病初起，津液微损。</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>3</label><question>针对图中的舌象应如何治疗？</question><answer>宜疏解透表并顾护津液。</answer><solution>若原文给出方药或治法，按原文摘录；无则留空不编造。</solution><figure_id>1.1.1</figure_id><source_text>id:144 治宜疏表透邪，酌加养阴护津之法。</source_text><pic>images/xxx....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图1-44、1-45、1-46 依次显示什么苔色？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图1-44 一条、图1-45 一条、图1-46 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<think>`/analysis/draft text.
"""
        PROMPT += """
输出前自检（必须通过）：
1) 每条 `<question>` 是否只覆盖一个焦点（特征/意义/治疗）？
2) `<question>` 是否都包含“图中/图中的”，且不含图号字符串？
3) `<answer>` 是否与题目焦点一致，且未扩写书外信息？
4) 若生成治疗题，`<solution>` 是否有可追溯原文依据？
5) `<source_text>` 是否均为「id:数字 + 原句」，且不含占位词？
6) `<figure_id>` 是否为纯数字点号格式（如 1.1.1），且与当前 `<pic>` 语义一致？若无法唯一确定是否已留空？
7) 是否完全没有多图合问、跨图取证、无依据治疗结论？
8) 是否根据输入证据选择了题型？有特征没出特征题、有治疗没出治疗题等遗漏情况判不合格。
"""
        return PROMPT

