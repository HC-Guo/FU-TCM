from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（《中医舌诊彩色图谱 中英文对照（龚一萍）》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 第一章 舌诊概说：舌的形态结构；舌诊原理和意义（舌与脏腑经络、气血津液）；舌诊方法与注意事项；舌诊内容与正常舌象（含生理变异）
- 第二章 望舌质：舌神（荣舌/枯舌）、舌色（淡红/淡白/红/绛/青紫）、舌形（老嫩、胖瘦、点刺、裂纹、齿痕）、舌动态（痿软/强硬/歪斜/颤动/吐弄/短缩）、舌下络脉
- 第三章 望舌苔：苔质（薄厚、润燥、腐腻、剥苔、偏全、真假）与苔色（白、黄、灰黑）
- 第四章 舌象分析要点及舌诊意义：察舌神气胃气、舌质舌苔综合分析、动态分析；临床意义（邪正、病邪性质、病位浅深、病势进退、预后）
- 图号格式：以原书印刷体为准，常见为「图1-1」「图1-3（1）」「图1-3（2）」等连字符样式
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
Document hierarchy hint (《中医舌诊彩色图谱 中英文对照（龚一萍）》; do not invent text): chapter flow is 舌诊概说 → 望舌质 → 望舌苔 → 舌象分析要点及舌诊意义; figure labels usually like 图X-X / 图X-X（1）. Prefer section titles as printed on the page.
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
<qa_pair><label>1</label><question>据图1-3（1）（以原书图号为准），本书如何概括该舌象或结构要点？<pic>tag2:box5</pic></question><answer>按该图图注给出结论（中文）。</answer><solution>结合邻文补充辨析或临床意义（中文）。</solution></qa_pair>
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
你是《中医舌诊彩色图谱 中英文对照（龚一萍）》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。本书主线为**第一章舌诊概说、第二章望舌质、第三章望舌苔、第四章舌象分析要点及舌诊意义**，并且常有**中英对照**。问答须紧扣中文原书条目命名与图注，**忽略英文平行翻译**，结果全部用中文。

**零容忍输出约束（优先级最高）**
- 你是**最终答案通道**，不是草稿通道。只允许输出最终 XML 结果。
- **严禁输出任何思考/草稿/解释文本**，包括但不限于：`<think>`、`<thinking>`、`<reasoning>`、`analysis`、`Chain-of-Thought`、`Now, ...`、`Let's ...`、"我先..." 等元叙述。
- 一旦输出任何上述内容，将导致下游解析失败并整批作废。请直接从 `<chapter>` 或 `<empty></empty>` 开始输出。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中的每个 `img_path`，再围绕该图在顺序上**附近**找文字（图注、上下行说明、同小节标题），生成 QA。
- **文字为辅**：不要为了凑条数先写大量无图 QA；本段内**每张有效教学图须输出多条带该图 `<pic>` 的 `<qa_pair>`（见下条数）**。
- **`id` 递增 ≈ 阅读顺序**，就近配对图注与正文。
- **语言约束（强制）**：`<title>/<question>/<answer>/<solution>/<source_text>` 统一使用中文；英文平行段不作为主要证据，不得产出英文问答。

{_TONGUE_BOOK_OUTLINE}

**核心任务：以图片为中心，为每张有意义的图片生成 3 类独立的 `<qa_pair>`：**
1) **特征类**：描述图中舌象的外观特征（如舌色、舌形、舌苔等）
2) **临床意义类**：说明该舌象提示什么证候或临床意义
3) **机理分析类**：阐述该舌象形成的病机（为什么会这样）

**三类问题的问法示例（可灵活变化，但须围绕主题）**

特征类问法（必须包含「图中舌象」）：
- 「图中舌象有何特征？」
- 「图中舌象有什么特征？」
- 「请描述图中舌象的形态特点。」

临床意义类问法（必须包含「图中舌象」）：
- 「图中舌象有何临床意义？」
- 「图中舌象提示什么？」
- 「图中舌象的主病是什么？」

机理分析类问法（必须包含「图中舌象」）：
- 「图中舌象形成的病机是什么？」
- 「为什么会出现图中舌象？」
- 「图中舌象的机理如何解释？」

**重要**：
- 三类题目尽量都出，但如果 `source_text` 中缺少某类信息（如只有特征没有机理），则只出有依据的类别。
- 结构/解剖类图片（如舌背分区、舌下络脉结构等）可以不生成临床意义和机理分析。
- 每类独立成一条 `<qa_pair>`，不得合并。

**图片处理规则（最重要）**
1. **每张图片多条 `<qa_pair>`（强制）**：输入 JSON 中每个有 `img_path` 且非装饰/页眉类的图片项，须生成 **多条** `<qa_pair>`，**每条**内**恰好一个** `<pic>` 且路径相同（特征、临床意义、机理分析三类，有证据就出，无证据不出）。若该图邻接原文极少、确实展不开，**不得少于 1 条**。
2. **一图多题拆分**：同一图下可将证型结论、病机阐释、鉴别要点、治法方药/调护、临床意义等**拆入不同 `<qa_pair>`**；**禁止**把多张不同 `img_path` 合并进同一 `<qa_pair>`。
3. **图文配对优先级**（从高到低）：
   - 图注行（如「图1-1 舌背各部名称」「图1-3（1） …」，以原书印刷图号为准）
   - 图片前最近的章节/小节标题
   - 图片后 1～3 行说明文字
   - 同章节内与图主题一致的「○…提示…」句、辨证要点或图谱条文
4. **`<pic>`**：必须使用输入 JSON 中该项的 **`img_path` 原文**（与 JSON 一致），放在 `<qa_pair>` 内；**可**放在 `<question>` 末尾或 `</source_text>`/`</solution>` 之后，整个 `<qa_pair>` 内**恰好** `<pic>` 出现一次。
5. **问句单图约束（强制）**：`<question>` 只能询问当前这一张图，禁止出现多图并提的语言（如“图1.1.32和图1.1.33”“图1.1.34、图1.1.35、图1.1.36”“分别/依次/对比上述两图/三图”）。一条 `<qa_pair>` 只能有一个图号语义主体。

**舌象措辞：图注优先（强制）**
- 图注行通常最可靠（如「图2.1.1 舌质红而少津：提示肝肾阴虚」「图1.1.5 淡红舌：正常舌色」）。**问句里的舌象描述应优先逐字或紧缩摘自该图对应图注**，再与「○…提示…」句核对；二者冲突时**以图注为准**。
- 若无独立图注但有「（图x-x）」夹在正文内，取该括号前后最短完整的舌象短语写入问句。
- 禁止把**其它条目**（邻近段落、同页它病）的舌象词贴到本图上。

**图文一致性（强制）**
- 本流水线以 JSON 文本输入为主、模型未必直接看到像素时：以该图对应 **图注 + 紧邻说明** 作为「肉眼所见」的**权威文字代理**，问句舌象须与之对齐，不得矛盾。
- `<question>` 所写舌象（**舌质/舌苔/津液/舌形/舌态** 等）**必须与**该 `<pic>` 在书中所附**图注及图意**一致，不得与图意矛盾。
- **禁止**仅依据离图很远的正文「想当然」改写字样：若图注写「舌质红而少津」，问句**不得**写成「苔黄」；若图注未出现「苔黄」，**不得**擅自加入。
- 若无法从图注/紧邻说明确定具体舌象用语：问句用 **「图中舌象」** 指代（可带疾病名与图号），**不要编造**细到苔色、津液等细节。
- `<answer>` 的证型/病机须与同图图注「提示……」及原文一致；`<solution>` 只收录**该书该图/该病节**下的治法方药，勿串页。

**问答生成（字段分工）**
- `<question>`：遵守上文**图注优先**与**图文一致**。若邻文有可复核图号，则在问句中写一次完整图号并保持原书格式（如「图1-1」「图1-3（1）」）；若邻文无图号，允许不用编号问法，**不要硬写/硬猜图号**。题首或题中均可（如「图1-3（1）…」「据图2-10 …」）。**禁止**在问句**末尾**或句中重复「看图」套话——图像已与题目一一对应，勿赘述：禁止「（见 图…）」「（见图…）」「请参考 图…」「请结合 图… 说明/回答」「请参照 图…」等**与题中已出现的同一图号**重复堆砌；若题首已含图号，**句尾不得**再出现同一图号的括号或「请参考」类提示。可参考 T1～T24 自然改写。图谱节、染苔节在合适时可点出「动态变化/染苔鉴别/病位病势判断」（**禁止**汇总整章的泛泛一问）。
- `<answer>`：**短答**——先给出与图注一致的舌象/结构结论或证型名；**并尽量**用**自己的话**压缩，且**至少融入一条**来自图注**以外**的信息：同一 JSON 块内该图**前后文**、或**相邻 id** 中与该图主题直接相关的病机句、鉴别句、临床意义句（若输入里确有）。**禁止** `<answer>` 仅为图注的逐字缩写或只加「（据图注）」而无新增信息——若输入除图注外无可融合句，也须**改写句式**并点明「见于教材该图」类教学化表述，避免与图注完全同形。
- `<solution>`：**学习性长段（与 `<answer>` 分工）**——从输入 JSON 中**检索并摘录**（允许适度紧缩，勿改写证型/方名核心字）：
  - **注意事项、正常舌象、温病常见舌象变化各型（理论/单项图谱）**：该图所在小节或同段中的**原文释义、鉴别要点、临床意义、与染苔或动态变化的提醒**等；图前后若有「○」提示以外的说明段落，**优先写入**。若该段确有此类文字，**不得**让 `<solution>` 空着只留 `<answer>` 抄图注；**若输入除图注外无可引用的邻接句**，`<solution>` 可留空，但 `<answer>` 仍须改写句式，避免与图注逐字相同。
  - **染苔、附录（斑疹白瘩、歌诀）及兼涉方药条文**：该图所属小节内与**该图谱类型或染苔类型**对应的 **辨证要点、治法、代表方、用药指征** 等；**若 JSON 中同段存在方药或条文原文，必须写入 `<solution>`**，不得省略；**方证须与图注及邻文一致**，不匹配则留空。
- **`<source_text>`（强制，供人工/程序校验）**：**每条** `<qa_pair>` 末尾（建议放在 `</solution>` 与 `<pic>` 之间）必须输出 `<source_text>…</source_text>`。内容须为从**输入本段 JSON** 各条目的 **`text` 字段、`list_items`、`image_caption` 等中逐字复制**的可见正文（允许为连贯阅读做**换行/分号**连接，**不得改写证型/方名用字**）。**写法规范**：按条标注 `id:数字` 后**必须紧跟该行在 JSON 中的原文整句或连续片段**（每条 id 后正文一般不少于 15 字，原文更短时全文照录）。**严禁**仅用占位语代替正文，例如单独写「图注全文」「同上」「见 id:xxx（略）」「邻接 text 一句」而不贴出汉字原文。**禁止**在 `<source_text>` 内编造书中没有的句子；若无邻文可引，则至少**完整照录**该图 `image_caption` / 图注字符串本身（逐字）。
- `<label>`：**本段输出内**从 1 起**全局连续递增**（一图多题时条数多，勿与上一输出段混号）；勿重复使用同一 `<label>` 绑定不同 `<question>`。

**防「只抄图注」（强制）**
- 输出须体现**教材教学价值**：除辨认图注表型外，还要让读者学到**相邻正文**中的病机、意义或疗法。
- 若某条 `<qa_pair>` 的 `<answer>` 读起来与图注几乎一模一样，视为不合格——须回到输入 JSON 补入邻近句或充实 `<solution>`。

**答案格式（建议，不强制）**
- `<answer>` 仍以核心结论为主，但鼓励「结论 + 一句依据/鉴别/意义」——依据须**摘自输入 JSON**，勿杜撰。
- `<solution>` 可与 `<answer>` 用空行语义区分：`<answer>` 偏「是什么/属何证」；`<solution>` 偏「为何重要、如何治、注意什么」。
- 若原文未给依据或疗法，`<solution>` 可留空，但不得用虚构方药凑数；宁可留空，不要配错。

**绝对约束（下游解析）**
1. 只输出 `<chapter>`…`</chapter>` 或 `<empty></empty>`；禁止 JSON、禁止任何思考标签（如 `<think>`、`<thinking>`、`<reasoning>`、`<redacted_thinking>`）。
2. `<title>`、`<question>`、`<answer>`、`<solution>`、`<source_text>` 为自然语言；`<source_text>` 内可带「id:数字」便于对照 JSON，但不要整段只输出 id 列表而无正文。
3. 禁止汇总型问句：如「该疾病对应的舌诊提示有哪些？」「本病舌诊要点有哪些？」。
4. **一图多题 + `<pic>` + `<source_text>`（强制）**：本段 JSON 中**每一个**有效 `img_path`（非装饰/页眉）须在输出中出现 3 次（特征类 + 临床意义类 + 机理分析类各 1 条）。若某类信息在 source_text 中缺失，可少于 3 次，但需有依据。每条 `<qa_pair>` 内**恰好一个** `<pic>`，且**必须**含 `<source_text>`。**禁止**多图合并在同一 `<qa_pair>`；**禁止**无 `<pic>` 的 `<qa_pair>`（有图时）。若本段**完全无图**，才可输出纯文字 `<qa_pair>` 或 `<empty></empty>`。
5. 禁止输出任何与最终 XML 无关的前后缀文字（如“下面是结果”“说明如下”“继续”等）；输出首字符必须是 `<`，且应为 `<chapter>` 或 `<empty></empty>`。

**禁止**
- 禁止把多张图合并进同一个 `<qa_pair>`。
- 禁止跳过本段 JSON 中应保留的教学图（仅装饰/页眉/纯重复可略）。
- 禁止编造不存在的 `img_path`。
- 禁止在 `<question>` 中出现图号。
- 禁止生成治疗、方药、调护、鉴别等其他类型题目。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文的目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；**一图须多条 `<qa_pair>`**；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>1.1 舌神——荣舌</title>
<qa_pair><label>1</label><question>图中舌象有何特征？</question><answer>舌色红润，运动灵活。</answer><solution>荣舌是舌有神气的表现，舌色红润鲜明，舌体活动自如。</solution><source_text>id:710 舌象特征：舌色红润，运动灵活。</source_text><pic>images/xxx.jpg</pic></qa_pair>
<qa_pair><label>2</label><question>图中舌象有何临床意义？</question><answer>提示舌有神气，属正常舌象或病情轻浅，预后良好。</answer><solution>舌神以\"红活\"为辨别要点，红润的舌色和灵活的运动表明气血充足。</solution><source_text>id:711 临床意义：舌有神气，正常舌象或病情轻浅，预后良好。</source_text><pic>images/xxx.jpg</pic></qa_pair>
<qa_pair><label>3</label><question>图中舌象形成的机理是什么？</question><answer>气血充足，阴阳调和，舌体得以濡养。</answer><solution>气血津液上荣于舌，脏腑功能正常，故舌色红润、活动灵活。</solution><source_text>id:702 舌神是对舌象特征的综合判断，以\"红活\"作为辨别要点。</source_text><pic>images/xxx.jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图1.1.44、图1.1.45、图1.1.46 依次显示什么苔色？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图1.1.44 一条、图1.1.45 一条、图1.1.46 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<think>`/analysis/draft text.
"""
        return PROMPT

