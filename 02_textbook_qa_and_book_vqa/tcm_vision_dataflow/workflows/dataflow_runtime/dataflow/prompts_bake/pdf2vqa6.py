from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（当前为《中医望诊与舌诊彩色图解》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 第一章～第三章：望诊基础知识、全身望诊、局部望诊（头面五官、四肢、皮肤、痰涎、小儿络脉等），多为示意图和局部照片
- 第四章 望舌：舌诊概说、望舌质（舌神/舌色/舌形/舌态）、望舌下络脉、望舌苔等，是本书舌诊理论与舌象图的集中部分
- 第五章 常见心血管疾病证候与舌象：真心痛、胸痹、心悸等分证型配套舌象图片与辨证说明，结构多为「病名→证型→舌象/症状/治法」
- 书中图片多配有「图X-X」或「图X-X（1）」式图号及简短图注
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
Document hierarchy hint (《中医望诊与舌诊彩色图解》; do not invent text): chapter flow is 望诊基础 → 全身望诊/局部望诊（头面五官、四肢、皮肤、痰涎、小儿络脉等） → 第四章 望舌（舌质/舌色/舌形/舌态/舌下络脉/舌苔等） → 第五章 常见心血管疾病证候与舌象（真心痛/胸痹/心悸等分证型配套舌象图片）。本书多数教学图片带有印刷图号（如「图4-1」「图5-3（2）」），但 JSON 中将以 `img_path` 标识。
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
<qa_pair><label>1</label><question>据图中舌象（本书该图无图号），本书如何概括该舌象或结构要点？<pic>tag2:box5</pic></question><answer>按该图图注给出结论（中文）。</answer><solution>结合邻文补充辨析或临床意义（中文）。</solution></qa_pair>
</chapter>

Please now process the provided page_n image and output your result.
"""
        return PROMPT


@PROMPT_REGISTRY.register()
class QAExtractPrompt(PromptABC):
    """MinerU content_list JSON → LLM：以图片为中心，每张有效教学图须生成 3～5 个 QA（硬性目标）。"""

    def __init__(self):
        pass

    def build_prompt(self) -> str:
        PROMPT = f"""
你是《中医望诊与舌诊彩色图解》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。全书前半部分为望诊基础与全身/局部望诊（头面五官、四肢、皮肤、痰涎、小儿络脉等）示意与病例图，第四章系统讲解舌诊（舌质、舌色、舌形、舌态、舌下络脉、舌苔），第五章配合常见心血管疾病证候与舌象（真心痛、胸痹、心悸等）。问答须紧扣中文原书条目命名与图注，结果全部用中文；在有舌象或具体病证配图的情况下，**以图为主、文字为辅** 组织 QA。

**零容忍输出约束（优先级最高）**
- 你是**最终答案通道**，不是草稿通道。只允许输出最终 XML 结果。
- **严禁输出任何思考/草稿/解释文本**，包括但不限于：`<think>`、`<thinking>`、`<reasoning>`、`analysis`、`Chain-of-Thought`、`Now, ...`、`Let's ...`、"我先..." 等元叙述。
- 一旦输出任何上述内容，将导致下游解析失败并整批作废。请直接从 `<chapter>` 或 `<empty></empty>` 开始输出。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中的每个 `img_path`，再围绕该图在顺序上**附近**找文字（图注、上下行说明、同小节标题）生成 QA。
- **文字为辅**：不要为了凑条数先写大量无图 QA；有图时每条 QA 必须绑定对应 `<pic>`。
- **`id` 递增 ≈ 阅读顺序**，就近配对图注与正文。
- **语言约束（强制）**：`<title>/<question>/<answer>/<solution>/<source_text>` 统一使用中文。

{_TONGUE_BOOK_OUTLINE}

**核心任务：以图片为中心，先判章/节，再按章节固定类别生成 QA（强制）**
1. **先判章节**：依据最近标题、图注图号、邻近文本确定该图所属章/节。
   - 必须判断到“章 + 节”粒度（如“第二章第一节”“第二章第二/三/四节”“第四章第一节/第二节起”）；未判清前不得出题。
2. **再按该章类别出题（只能用下列类别，不得新增）**：
   - 第一章：跳过，不生成 QA。
   - 第二章第一节（图2-1～图2-8）：`临床意义`、`临床表现`。
   - 第二章第二/三/四节（图2-9～图2-51）：`什么表现`、`提示什么病症`（两类非强制，有证据才出）。
   - 第三章：`什么表现`、`提示什么病症`（两类非强制，有证据才出）。
   - 第四章第一节：跳过，不生成 QA。
   - 第四章第二节起（图4-3 起）：`舌象特征`、`临床意义`。
   - 第五章治疗前图片：`舌象特征`、`中/西医诊断`、`治则治法`。
   - 第五章治疗后图片：仅 `舌象特征`。
3. **问句只可同义改写**：不要求逐字一致，但语义必须与上述类别一一对应；不得新增示例外术语/问题类型。
4. **证据就近绑定**：`source_text` 仅可使用当前图号锚点前后 3 句，且必须与当前 `<pic>` 直接相关。
5. **有证据才生成（强制）**：某类别只有在 `image_caption`、邻近 `text`、`list_items` 中存在直接证据时才可生成；无证据可不生成，不得臆造。
6. **不确定就跳过（强制）**：若无法可靠判定当前图所在节，或无法为候选类别找到直接证据，则该类别不生成；必要时整图跳过。

**图片处理规则（最重要）**
1. **每张图片多条 `<qa_pair>`（强制）**：输入 JSON 中每个有 `img_path` 且非装饰/页眉类的图片项，须生成 **3～5 条** `<qa_pair>`；每条内**恰好一个** `<pic>` 且路径相同。
2. **一图多题拆分**：同一图下不同类别分开成题；禁止把多张不同 `img_path` 合并进同一 `<qa_pair>`。
3. **图文配对优先级**（从高到低）：
   - 图注行/图片说明文字（通常包含「图X-X…」或「图X-X（1）」式图号及舌象/病变描述）
   - 图片前最近的章节/小节标题
   - 图片后 1～3 行说明文字
   - 同章节内与图主题一致的「○…提示…」句、辨证要点或图谱条文（望诊要点、舌象解读、心血管疾病各证型说明等）
4. **`<pic>`**：必须使用输入 JSON 中该项的 **`img_path` 原文**（与 JSON 一致），放在 `<qa_pair>` 内；**可**放在 `<question>` 末尾或 `</source_text>`/`</solution>` 之后，整个 `<qa_pair>` 内**恰好** `<pic>` 出现一次。
5. **问句单图约束（强制）**：`<question>` 只能询问当前这一张图，禁止出现多图并提的语言（如“两图/三图”“分别/依次/对比上述两图”）。一条 `<qa_pair>` 只能有一个图片语义主体。
6. **问句图号规则（强制）**：
   - `<question>` 中**禁止**出现任何图号或其变体（如「图2-1」「图 2-1」「图2.1」「Fig.2-1」）。
   - 统一使用无图号表达：如「图中舌象」「该图表现」「该图所示」。
   - 若原文只靠图号才能定位，也必须改写为无图号单图问法；无法改写则丢弃该条。
7. **单图防串图（强制）**：`<question>` 不得出现第二个图号或跨图比较语（如“前图/后图/两图/共同提示/分别/依次/对照”）；命中即重写，无法重写则删除。

**舌象措辞：图注优先（强制）**
- 图注行通常最可靠（如「图2.1.1 舌质红而少津：提示肝肾阴虚」「图1.1.5 淡红舌：正常舌色」）。**问句里的舌象描述应优先逐字或紧缩摘自该图对应图注**，再与「○…提示…」句核对；二者冲突时**以图注为准**。
- 若无独立图注但有「（图x-x）」夹在正文内，取该括号前后最短完整的舌象短语写入问句。
- 禁止把**其它条目**（邻近段落、同页它病）的舌象词贴到本图上。

**图文一致性（强制）**
- 本流水线以 JSON 文本输入为主、模型未必直接看到像素时：以该图对应 **图注 + 紧邻说明** 作为「肉眼所见」的**权威文字代理**，问句舌象须与之对齐，不得矛盾。
- `<question>` 所写舌象（**舌质/舌苔/津液/舌形/舌态** 等）**必须与**该 `<pic>` 在书中所附**图注及图意**一致，不得与图意矛盾。
- **禁止**仅依据离图很远的正文「想当然」改写字样：若图注写「舌质红而少津」，问句**不得**写成「苔黄」；若图注未出现「苔黄」，**不得**擅自加入。
- 若无法从图注/紧邻说明确定具体舌象用语：问句用 **「图中舌象」** 指代（可带疾病名，不带图号），**不要编造**细到苔色、津液等细节。
- `<answer>` 的证型/病机须与同图图注「提示……」及原文一致；`<solution>` 只收录**该书该图/该病节**下的治法方药，勿串页。

**问句模板（强约束：可改写，不可改义）**
- 第二章第一节：`图中可见哪些临床表现？` / `该图表现提示什么临床意义？`
- 第二章第二、三、四节：`该图主要表现是什么？` / `该图表现提示什么病症？`
- 第三章：`图中主要表现如何概括？` / `根据该图可提示什么病症？`
- 第四章第二节起：`该图舌象特征如何概括？` / `该舌象在临床上有什么意义？`
- 第五章治疗前：`该图舌象特征是什么？` / `根据该图可支持怎样的中/西医诊断？` / `针对该图表现应如何把握治则治法？`
- 第五章治疗后：仅 `该图舌象特征是什么？`
- 除上述同义问句外，禁止输出其它问题类型。

**类别缺证据时的处理（强制）**
- 以上类别不是机械配额；某类若无直接证据可不生成。
- 但某类若在 `image_caption`、邻近 `text`、`list_items` 中存在直接证据，则应生成该类 QA。
- **程序友好判定**：按“章节允许类别清单”逐类检索证据，有则产出、无则跳过；禁止臆造。

**证据窗口约束（强制，确保图文强相关）**
- 每条 QA 必须先定位当前图片的图号锚点（如“图X-X”或该图对应 `image_caption` 所在句）。
- `<source_text>` 仅允许引用该锚点**前 3 句到后 3 句**范围内的原文句子（含图注句本身）；超出该窗口的句子一律不得使用。
- 若同段含多个图号，必须选择与当前 `<pic>` / `figure_id` 一致的锚点窗口，禁止借用其它图号窗口内容。
- `<answer>` 与 `<solution>` 的核心结论必须可由该窗口内句子直接支持；若窗口内证据不足，则留空或跳过，不得外推。
- `<source_text>` 中每个 `id` 片段都必须与当前图直接相关；若出现仅描述其它图片、其它病例或跨节总述的句子，必须删除。

**问句改写边界（强制）**
- 可做轻量同义改写（如“概括/描述/提示”互换），但语义必须与模板一致。
- 禁止新增术语、禁止新增问题意图、禁止越过章节类别边界。
- 仅在证据充分时改写生成；证据不足时不得为了凑条数硬写。

**问答生成（字段分工）**
- `<question>`：遵守上文**图注优先**与**图文一致**。本书问句统一使用无编号问法（如「图中舌象…」「该图表现…」），**不要输出任何图号**。**禁止**在问句末尾或句中重复「看图」套话（如「见图/请参考该图」）。问句只能是上文示例类别的同义改写，严禁输出任何未示例的问题类型。
- `<answer>`：**短答**——先给出与图注一致的舌象/结构结论或证型名；**并尽量**用**自己的话**压缩，且**至少融入一条**来自图注**以外**的信息：同一 JSON 块内该图**前后文**、或**相邻 id** 中与该图主题直接相关的病机句、鉴别句、临床意义句（若输入里确有）。**禁止** `<answer>` 仅为图注的逐字缩写或只加「（据图注）」而无新增信息——若输入除图注外无可融合句，也须**改写句式**并点明「见于教材该图」类教学化表述，避免与图注完全同形。
- `<solution>`：**学习性长段（与 `<answer>` 分工）**——从输入 JSON 中检索并摘录（允许适度紧缩）：
  - 第二章：补充临床表现、临床意义相关原文；
  - 第三章：补充“什么表现/提示什么病症”的原文证据；
  - 第四章（第二节起）：补充舌象特征、临床意义相关证据；
  - 第五章：若同段有中/西医诊断、治则治法、方药或调护原文，应写入 `<solution>`；
  - 无证据可留空，禁止编造。
- **弱答案禁用词（强制）**：`<answer>` 与 `<solution>` 禁止出现占位语或空泛兜底语，包括但不限于：`对应证型`、`需结合上下文`、`以图注为准`、`见原文`、`同上`、`略`。命中则必须改写为可验证结论；若无足够证据可改写，允许留空但不得保留占位语。
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
4. **一图多题 + `<pic>` + `<source_text>`（强制）**：本段 JSON 中**每一个**有效 `img_path`（非装饰/页眉）须在输出中出现 **至少 3 次、通常 3～5 次**（每条含该路径的 `<qa_pair>` 各一次 `<pic>`），每条 `<qa_pair>` 内**恰好一个** `<pic>`，且**必须**含 `<source_text>`。**同一图仅出现 1 次 `<pic>` 的产出视为错误，必须重写到 ≥3 次。** **禁止**多图合并在同一 `<qa_pair>`；**禁止**无 `<pic>` 的 `<qa_pair>`（有图时）。若本段**完全无图**，才可输出纯文字 `<qa_pair>` 或 `<empty></empty>`。
5. 禁止输出任何与最终 XML 无关的前后缀文字（如“下面是结果”“说明如下”“继续”等）；输出首字符必须是 `<`，且应为 `<chapter>` 或 `<empty></empty>`。
6. `question` 文本中禁止“多图合问”模式：若出现“和/及/、”连接多个图片对象（如“前图和后图”“两图对比”），视为不合格，必须拆成多条单图 QA。

**第四章非舌象图过滤（强制）**
- 第四章中若图片为穴位定位、经络示意、按压操作、纯说明图标等非舌象主题，跳过不生成 QA。
**证型与治法匹配（重要提醒）**
- 第五章涉及中/西医诊断与治则治法时，结论必须与同图证据一致；不确定可留空，不可硬配。

**禁止**
- 禁止把多张图合并进同一个 `<qa_pair>`。
- 禁止跳过本段 JSON 中应保留的教学图（仅装饰/页眉/纯重复可略）。
- **禁止「一图一题交差」**：对有效教学图输出仅含该图 **1 条** `<qa_pair>` 视为未完成任务，须在同一输出中扩写到 **≥3 条**。
- 禁止编造不存在的 `img_path`。
- **问句禁冗余指图**：`<question>` 内禁止叠用「见图/参照图/结合图」类提示；一句一问，指图一次即可。
- **禁止多图合问**：禁止在同一 `<question>` 中出现两个及以上图片对象并列表述；必须拆成单图 QA。
- **禁止单图跨图比较**：若 `<images>` 只有 1 张，题干不得写“与另一图相比”“与前图对照”等跨图比较词；必须改写为仅围绕当前图的问题，否则删除该条。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文的目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；同图可生成多条 `<qa_pair>`；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>第二章 第一节</title>
<qa_pair><label>1</label><question>图中可见哪些临床表现？</question><answer>根据图注可见舌象相关表现为……</answer><solution>补充邻文中的表现描述与判读要点。</solution><source_text>id:101 ……；id:102 ……</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>该图表现提示什么临床意义？</question><answer>提示……</answer><solution>结合同段原文解释其临床意义。</solution><source_text>id:102 ……；id:103 ……</source_text><pic>images/xxx....jpg</pic></qa_pair>
</chapter>

<chapter><title>第五章 某证候节</title>
<qa_pair><label>3</label><question>该图舌象特征如何概括？</question><answer>舌象特征为……</answer><solution>摘录同段舌质舌苔描述。</solution><source_text>id:201 ……</source_text><pic>images/yyy....jpg</pic></qa_pair>
<qa_pair><label>4</label><question>根据该图可支持怎样的中/西医诊断？</question><answer>可支持……诊断。</answer><solution>摘录同段中医/西医诊断依据。</solution><source_text>id:202 ……</source_text><pic>images/yyy....jpg</pic></qa_pair>
<qa_pair><label>5</label><question>针对该图表现应如何把握治则治法？</question><answer>治则治法宜……</answer><solution>摘录同段治法方药或调护原文。</solution><source_text>id:203 ……</source_text><pic>images/yyy....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图2-1和图2-2分别提示什么？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图A 一条、图B 一条、图C 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<redacted_thinking>`/analysis/draft text.
Reminder: follow chapter-category rules strictly, only use allowed category semantics, keep questions figure-number-free, and do not apply image size/area filtering.
"""
        return PROMPT

