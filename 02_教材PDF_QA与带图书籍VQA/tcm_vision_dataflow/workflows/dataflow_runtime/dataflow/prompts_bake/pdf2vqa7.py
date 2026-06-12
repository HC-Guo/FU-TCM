from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（《望舌诊病与中医简易治疗》类教材），供模型对齐章节层级
_TONGUE_BOOK_OUTLINE = """
Book outline (approx. 135 pages; use only to recognize hierarchy, not to invent text):
- 第一部分 望舌诊病概述
  - 一、舌诊基础知识 — (一)舌的形态特点 (二)舌的组织结构 (三)舌诊原理 (四)舌诊的意义 (五)舌诊方法
  - 二、各种舌象所反映的疾病 — (一)舌质 (二)舌苔 (三)舌脉
- 第二部分 常见病舌诊要点及中医简易治疗
  - 一、传染病与寄生虫病 — (一)病毒性肝炎 (二)肺结核 (三)钩虫病
  - 二、呼吸系统疾病 — (一)急性支气管炎 (二)慢性支气管炎 (三)支气管哮喘 (四)支气管扩张
  - 三、消化系统疾病 — (一)急性胃炎 … (八)胆石症
  - 四、心脑血管疾病 — … (九)原发性高血压
  - 五、内分泌疾病和代谢性疾病 — (一)甲亢 (二)围绝经期综合征 (三)糖尿病 (四)痛风
  - 六、神经系统疾病 — (一)神经衰弱 (二)头痛
  - 七、泌尿系统疾病 — …
  - 八、妇科病和男科疾病 — …
  - 九、运动系统疾病 — …
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
Strict extraction rules:
** Units (questions / answers / solutions) **
- If the page is cover, copyright, pure catalog, page numbers only, or irrelevant front/back matter, output `<empty></empty>`.
- **Labels**: preserve the book’s markers, e.g. "（一）", "（二）", "一、", "1.", "例1". Prefer Arabic digits only when the book mixes styles (例一→例1). For headings like "（三）钩虫病", use that as label for the following qa_pair(s).
- Multiple sub-points (1)(2) or (1)(a) under one disease block: keep them in **one** `<qa_pair>`…`</qa_pair>`.
- If 舌象、辨证、治法、方药 are contiguous, one `<qa_pair>` with question/solution split sensibly; if clearly separated by headings, use separate `<qa_pair>` blocks with appropriate labels.
- Use `<answer>` for very short outcomes (e.g. 治则一词) when the book separates them; use `<solution>` for longer 方药、煎服法、注意事项.
** Chapter / section titles (text in <title>, this prompt mode) **
- `<chapter>`…`</chapter>` with `<title>…</title>` = the section heading text on the page (e.g. "（一）病毒性肝炎" or "二、呼吸系统疾病"). Multiple chapters if multiple such headings appear.
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
<chapter><title>（一）病毒性肝炎</title>
<qa_pair><label>1</label><question>舌象特征与辨证要点。<pic>tag2:box5</pic></question><answer>肝胆湿热兼脾虚等（据原文）。</answer><solution>简易治疗与调护（据原文）。</solution></qa_pair>
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
你是《望舌诊病与中医简易治疗》类教材的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。

**零容忍输出约束（优先级最高）**
- 你是**最终答案通道**，不是草稿通道。只允许输出最终 XML 结果。
- **严禁输出任何思考/草稿/解释文本**，包括但不限于：`<think>`、`<thinking>`、`<reasoning>`、`analysis`、`Chain-of-Thought`、`Now, ...`、`Let's ...`、"我先..." 等元叙述。
- 一旦输出任何上述内容，将导致下游解析失败并整批作废。请直接从 `<chapter>` 或 `<empty></empty>` 开始输出。

**范式转变（必须遵守）**
- **以图为主、文字为辅**：先枚举本段 JSON 中每个 `img_path`，再围绕该图就近取证（图注、邻近说明、同小节标题）生成 QA；禁止先写无图泛化题凑数。
- **`id` 递增 ≈ 阅读顺序**，按顺序就近配对图注与正文。

{_TONGUE_BOOK_OUTLINE}

**核心任务：以图片为中心，为每张有意义的图片生成独立 `<qa_pair>`；优先多题，但以证据充分为前提（宁少勿错）。**

**题目白名单（最高优先级，强制）**
- 不再自由生成问句；`<question>` 必须从以下**固定题型**中选择并仅做最小槽位填充（维度词）：
  1) 「图中的[舌色/舌形/舌态]是什么？」
  2) 「图中的[舌象/舌色/舌形/舌态]有什么特征？」
  3) 「图中的[舌色/舌形/舌态]提示了什么？」
  4) 「图中的舌苔是什么[苔质/苔色]？」
  5) 「图中的舌苔提示了什么？」
  6) 「图中的舌象有什么特征？」
  7) 「图中的舌象提示了什么？」
- 每条问句只能对应一个焦点：若使用第 6 条“舌象有什么特征”题型，答案可按原文列出多维特征；其它题型仍是一题一焦点。
- 同一张图优先按以下顺序出题：先识别（是什么）→再特征（有哪些特征）→再提示（提示了什么）；仅在原文有依据时保留舌苔相关题型。
- **语义覆盖最小集（强制）**：对同一张图，若同号证据中同时存在“命名/定义信息”（如“称为…舌”）与“可见特征信息”（如“纹理粗糙/淡白胖嫩/苔白滑/少津”等），则至少输出 2 条 QA：一条“是什么”（命名），一条“有什么特征”（特征）；不得只输出“名称 + 提示”而缺少特征题。
- **信息落题约束（强制）**：若同号证据或 `solution` 可抽取到明确“提示/主病/病机”信息（如“提示…/多为…/主…”），不得只写在 `solution`；必须至少增加 1 条对应问句（优先「图中的舌象提示了什么？」或「图中的舌苔提示了什么？」）。禁止出现“question 未覆盖、solution 单独扩写”的信息孤岛。
- 严禁输出白名单外问句（包括“如何调护/饮食疗法有哪些/如何鉴别/临床意义为何”等自由问法）。
- **模板槽位必填（强制）**：题型中的方括号占位符必须在生成前全部替换为具体词，禁止把占位符原样输出到最终 `<question>`。尤其「图中的舌苔是什么[苔质/苔色]？」必须二选一填成「图中的舌苔是什么苔质？」或「图中的舌苔是什么苔色？」。
- **舌苔题型触发门槛（强制）**：仅当证据句中出现明确苔维度词（如「苔/苔质/苔色/薄/厚/腻/腐/剥/无苔/少苔/滑苔/燥苔」）时，才允许生成“舌苔”题型（包括「舌苔是什么苔质/苔色」「舌苔提示了什么」）。
- **非苔维度禁用舌苔题型（强制）**：若证据句仅出现「舌面润滑/舌体胖嫩/舌质色青」等舌质、舌形、舌态表述，且无明确苔维度词，禁止生成舌苔题型；必须改用「图中的舌象有什么特征？」或「图中的舌象提示了什么？」等非苔题型。

**结构/解剖图专用白名单（第一部分可用，强制）**
- 当图片属于舌体结构、分区、解剖定位（如舌尖/舌中/舌根/舌边、舌下阜、金津玉液、脏腑分区）时，`<question>` 只能使用以下固定句式：
  1) 「图中标示的结构是什么？」
  2) 「图中展示的结构有哪些特征？」
  3) 「图中所示分区提示了什么？」
- 结构/解剖图不得使用“舌色/舌形/舌态/舌苔”题型；舌象图也不得反向使用结构题型。

**图片处理规则（最重要）**
1. **每张图片条数与证据匹配（强制）**：输入 JSON 中每个有 `img_path` 且非装饰/页眉类的图片项，优先生成 **2～5 条** `<qa_pair>`；若同号锚点句可支撑的信息不足，允许降到 **1 条**。禁止为凑条数跨句、跨图、跨段扩写。每条内**恰好一个** `<pic>` 且路径相同；各条 `<question>` 必须从“题目白名单”中选取。
2. **一图多题拆分**：同一图按图类型拆分——舌象图使用“题目白名单”7句式，结构/解剖图使用“结构/解剖图专用白名单”3句式；**禁止**把多张不同 `img_path` 合并进同一 `<qa_pair>`。
2.1 **细粒度拆题（强制）**：同一图若文本同时出现舌色/舌形/舌态/苔质/苔色/津液等信息，必须拆成多条题目，遵循“**一题一维度**”。例如：舌色单独一题、舌形单独一题、舌态单独一题、苔质或苔色单独一题、整体提示（病机/证型）再单独一题。禁止把多个维度揉进同一问句。
2.2 **题干焦点唯一（强制）**：每条 `<question>` 只问一个核心点（如“是什么”“有何特征”“提示什么”三者选一），不得在一题里并列“是什么+特征+提示了什么”。
2.3 **细粒度反例（强制禁止）**：严禁出现「图2-6 的舌苔苔色和苔质如何描述」「图2-8 的舌质和舌苔分别呈现什么特征」这类并列双维度/多维度问句。必须拆成两条及以上单点题（如苔色一题、苔质一题、舌质一题）。
2.4 **并列词触发拆题（强制）**：若问句中出现“和/及/与/分别/同时”等并列词，且所连对象属于不同维度（如舌质 vs 舌苔、苔色 vs 苔质、舌色 vs 舌形），必须拆题；不得保留为一条。
3. **图文配对优先级**（从高到低）：
   - 图注行（如「图1-1 舌背各部名称」「图2-1 …」）
   - 图片前最近的章节/小节标题
   - 图片后 1～3 行说明文字
   - 同章节内与图主题一致的「○…提示…」句（疾病篇）
3.1 **caption 延伸纠偏（强制）**：若 `image_caption`/图注包含多句延伸说明、并非都严格对应该图，必须先定位含该图号的锚点文本（如「（图2-6）」或「图2-6 …」）作为主依据；无同号锚点时仅取紧邻该锚点前后一句，不得整段照搬为该图结论。
3.2 **同号优先匹配（强制）**：同一 JSON 段内出现多个图号时，当前 `<pic>` 只能引用与本图同号（图M-N）的句段；出现冲突时舍弃不带同号锚点的泛化描述。
3.3 **`<source_text>` 截取边界（强制）**：`<source_text>` 只允许包含与当前 `img_path` 同号锚点（图M-N）紧邻的一句描述性原文；禁止纳入该图所在小节的病因总论、病机归纳或治疗通论性文字。
3.4 **图注主句锁定（强制）**：当同一图号附近同时出现“主定义句 + 若…则/若…多提示…/变体条件句”时，默认只用图注/同号锚点的主定义句；不得用后续分支句替代主描述。
3.5 **QA 取句范围硬约束（最高优先级，强制）**：`<question>/<answer>/<source_text>` 证据只能来自当前 `<pic>` 的同号句集合：至少包含 1 条同号锚点句（图M-N），并可补充其相邻位置中**同样显式含该图号**的句子；禁止使用不含该图号的前后句，禁止跨段、跨图取证。
3.5.1 **首证据句门槛（强制）**：`<source_text>` 的第一条证据必须是显式同号锚点句（含“图M-N”或“（图M-N）”）或该图 `image_caption` 原文；否则整条 `<qa_pair>` 判不合格并重写。
3.6 **无图号句禁用（最高优先级，强制）**：不含当前图同号锚点（如“图1-15”或“（图1-15）”）的句子不得用于该图。尤其“若见…则…”句若无同号图号，一律禁用；若 `<source_text>` 出现“若见/若…则/若…多提示”且该分句不含同号图号，直接判不合格。
3.7 **无同号锚点时的降级策略（强制）**：若找不到同号锚点句，只允许使用该图 `image_caption`（或图片项同条 `text`）生成 1 条最保守 QA；仍不足则该图不生成（宁缺毋滥）。
3.8 **同号邻句优先补充（强制）**：若图注之外在邻近文本中还能找到显式同号句（含“图M-N”或“（图M-N）”），应优先纳入 `<source_text>` 作为补充证据；不得因为“只取图注”而忽略这些同号句。
3.9 **非图注同号句强制纳入（强制）**：若存在“非 image_caption 的显式同号句”，则该图至少 1 条 `<qa_pair>` 的 `<source_text>` 必须同时包含「图注句 + 该非图注同号句」或直接使用该非图注同号句；仅用图注且忽略可用同号邻句，判不合格。
3.10 **多图号同句切片规则（强制）**：当同一原句同时出现多个图号（如“...图1-32...图1-33...”），允许按当前 `figure_id` 进行最小切片引用；切片后文本必须仍含当前图号并语义完整。禁止因“同句含多图号”而整句弃用。
4. **`<pic>`**：必须使用输入 JSON 中该项的 **`img_path` 原文**（与 JSON 一致），放在 `<qa_pair>` 内；**可**放在 `<question>` 末尾或 `</source_text>`/`</solution>` 之后，整个 `<qa_pair>` 内**恰好** `<pic>` 出现一次。
5. **问句单图约束（强制）**：`<question>` 只能询问当前这一张图，禁止出现多图并提的语言（如“图1-32和图1-33”“图1-34、1-35、1-36”“分别/依次/对比上述两图/三图”）。一条 `<qa_pair>` 只能有一个图号语义主体。
6. **问句须含“图中”锚点（强制）**：每条 `<question>` 必须明确出现“图中”或“图中的”，禁止脱离图片直接提问。
6.1 **禁止脱图定义题（强制）**：禁止生成可脱离图片成立的定义题（如“淡白湿润胖嫩舌有哪些伴随特征？”）。必须改写为“图中……”。 

**舌象措辞：图注优先（强制）**
- 图注行通常最可靠（如「图2-1 舌质红而少津：提示肝肾阴虚」「图1-5 淡红舌：正常舌色」）。**问句里的舌象描述应优先逐字或紧缩摘自该图对应图注**，再与「○…提示…」句核对；二者冲突时**以图注为准**。
- 若无独立图注但有「（图x-x）」夹在正文内，取该括号前后最短完整的舌象短语写入问句。
- 禁止把**其它条目**（邻近段落、同页它病）的舌象词贴到本图上。
- 若同图后续出现“若…则…”条件分支（如“若淡红舌…为淡白夹红舌”），该分支默认视为**派生亚型**，不得覆盖图注主名词（除非题干明确写“淡白夹红舌/变体”）。

**图文一致性（强制）**
- 本流水线以 JSON 文本输入为主、模型未必直接看到像素时：以该图对应 **图注 + 紧邻说明** 作为「肉眼所见」的**权威文字代理**，问句舌象须与之对齐，不得矛盾。
- `<question>` 所写舌象（**舌质/舌苔/津液/舌形/舌态** 等）**必须与**该 `<pic>` 在书中所附**图注及图意**一致，不得与图意矛盾。
- **禁止**仅依据离图很远的正文「想当然」改写字样：若图注写「舌质红而少津」，问句**不得**写成「苔黄」；若图注未出现「苔黄」，**不得**擅自加入。
- 若无法从图注/紧邻说明确定具体舌象用语：问句用 **「图中舌象」** 指代（可带疾病名与图号），**不要编造**细到苔色、津液等细节。
- `<answer>` 的证型/病机须与同图图注「提示……」及原文一致；`<solution>` 只收录**该书该图/该病节**下的治法方药，勿串页。

**问句生成方式（强制）**
- 仅允许使用上方白名单固定句式；不做自由改写。
- 先判图类型（结构/解剖图 vs 舌象图），再从对应白名单选句式；禁止跨白名单混用。

**疾病篇（第二部分）问句（强制）**
- 只能使用“题目白名单”句式；禁止“属何证/如何调护/如何鉴别”等白名单外问法。
- 治法方药可写入 `<solution>`，不得作为 `<question>` 提问目标。

**问句示例池（已禁用）**
- 为避免模型漂移，示例池（A/T/D）不再生效；实际生成仅允许“题目白名单”7种句式。

**问答生成（字段分工）**
- `<question>`：必须严格使用“题目白名单”固定句式并填入维度词；不得自行改写、扩写或生成白名单外问法。仍需满足“图中”锚点、单图、图文一致与一题一焦点约束。
- `<question>` **覆盖补充（强制）**：若同号证据含“称为/是…舌”等命名句，且同号证据另有可直接抽取的形态描述句（特征短语），必须补一条“图中的…有什么特征？”；禁止仅保留命名题与提示题。
- `<answer>`：**短答**——先给出与图注一致的舌象/结构结论或证型名；可用自己的话压缩。来自图注以外的信息（前后文病机/鉴别/临床意义）仅作**可选补充**，不得反客为主、不得改变图注主结论。允许 `<answer>` 仅依据图注主句作答（尤其第一部分定义类图），不再强制追加“图注外信息”。
- `<answer>` **维度强匹配（强制）**：必须与 `<question>` 的提问维度一一对应，禁止跨维度混答。
  - 问「舌色是什么/有什么特征/提示了什么」：`<answer>` 只写舌色结论或由舌色直接导出的提示，禁止混入舌形/舌态/舌苔信息。
  - 问「舌形是什么/有什么特征/提示了什么」：`<answer>` 只写舌形结论或由舌形直接导出的提示，禁止混入舌色/舌苔/津液描述。
  - 问「舌态是什么/有什么特征/提示了什么」：`<answer>` 只写舌态结论或由舌态直接导出的提示，禁止混入舌色/舌形/舌苔信息。
  - 问「舌苔是什么苔质/苔色/舌苔提示了什么」：`<answer>` 只写舌苔维度，苔质与苔色不得在同一条答案中混写（除非问题本身就是“舌苔有什么特征”）。
  - 问「舌象有什么特征/舌象提示了什么」：才允许多维组合描述。
- `<answer>` 补充约束：**禁止**输出“暂无明确简易疗法记录”“可参照…调护原则”“可酌情处理”等无原文依据的兜底句。若输入 JSON 未提供该图对应治法，保留治疗题为空或不生成治疗题，勿编写占位结论。
- `<solution>`：**学习性长段（与 `<answer>` 分工）**——从输入 JSON 中**检索并摘录**（允许适度紧缩，勿改写证型/方名核心字）：
  - **第一部分（理论/舌象图谱）**：该图所在小节或同段中的**原文释义、鉴别要点、临床意义、注意事项**等；图前后若有「○」提示以外的说明段落，**优先写入**。若该段确有此类文字，**不得**让 `<solution>` 空着只留 `<answer>` 抄图注；**若输入除图注外无可引用的邻接句**，`<solution>` 可留空，但 `<answer>` 仍须改写句式，避免与图注逐字相同。
  - **第二部分（疾病与各证型图）**：以该图对应舌象的原文解释为主，允许在 `<solution>` 补充同段可追溯原文；不得反向驱动生成“治疗/调护/鉴别”类问句。
- **`<source_text>`（强制，供人工/程序校验）**：**每条** `<qa_pair>` 末尾（建议放在 `</solution>` 与 `<pic>` 之间）必须输出 `<source_text>…</source_text>`。内容须为从**输入本段 JSON** 各条目的 **`text` 字段、`list_items`、`image_caption` 等中逐字复制**的可见正文（允许为连贯阅读做**换行/分号**连接，**不得改写证型/方名用字**）。**写法规范**：按条标注 `id:数字` 后**必须紧跟该行在 JSON 中的原文整句或连续片段**（每条 id 后正文一般不少于 15 字，原文更短时全文照录）。**严禁**仅用占位语代替正文，例如单独写「图注全文」「同上」「见 id:xxx（略）」「邻接 text 一句」而不贴出汉字原文。**禁止**在 `<source_text>` 内编造书中没有的句子；若无邻文可引，则至少**完整照录**该图 `image_caption` / 图注字符串本身（逐字）。
- `<solution>` **不越权（强制）**：`<solution>` 不得新增未被任何 `<question>` 覆盖的核心事实（尤其“提示/主病/病机”）。若写入了该类事实，必须同步存在对应 QA；否则删去该事实或补题后再输出。
- **`<source_text>` 占位词零容忍（强制）**：禁止出现任何占位词或笼统指代，包括但不限于「图注原文」「第N条原文」「原文」「见原文」「同上原文」。`<source_text>` 必须是可核验的真实文本，格式固定为「`id:数字` + 原句」。若无法提供真实原句，该 `<qa_pair>` 判不合格并重写，直至 `<source_text>` 可追溯。
- **`<source_text>` 图号一致性（强制）**：当 `<figure_id>`/当前 `<pic>` 已确定图号时，`<source_text>` 必须包含同号锚点（如“图M-N”或“（图M-N）”）或直接来自该图图片项的 `image_caption` 原文；禁止引用不含该图号的泛化句（尤其“若见…则…”句）。
- `<label>`：**本段输出内**从 1 起**全局连续递增**（一图多题时条数多，勿与上一输出段混号）；勿重复使用同一 `<label>` 绑定不同 `<question>`。

**答案格式（建议，不强制）**
- `<answer>` 仍以核心结论为主，但鼓励「结论 + 一句依据/鉴别/意义」——依据须**摘自输入 JSON**，勿杜撰。
- `<solution>` 可与 `<answer>` 用空行语义区分：`<answer>` 偏「是什么/属何证」；`<solution>` 偏「为何重要、如何治、注意什么」。
- 若原文未给依据或疗法，`<solution>` 可留空，但不得用虚构方药凑数；宁可留空，不要配错。

**绝对约束（下游解析）**
1. 只输出 `<chapter>`…`</chapter>` 或 `<empty></empty>`；禁止 JSON、禁止任何思考标签（如 `<think>`、`<thinking>`、`<reasoning>`、`<redacted_thinking>`）。
2. `<title>`、`<question>`、`<answer>`、`<solution>`、`<source_text>` 为自然语言；`<source_text>` 内可带「id:数字」便于对照 JSON，但不要整段只输出 id 列表而无正文。
3. 禁止汇总型问句：如「该疾病对应的舌诊提示有哪些？」「本病舌诊要点有哪些？」。
4. **一图一题起步 + `<pic>` + `<source_text>`（强制）**：本段 JSON 中**每一个**有效 `img_path`（非装饰/页眉）须在输出中出现 **至少 1 次**（证据充分时可到 2～5 次）；每条 `<qa_pair>` 内**恰好一个** `<pic>`，且**必须**含 `<source_text>`。**同一张图的多条 `<qa_pair>` 必须各自独立（图号可相同，但 `<label>`、`<question>`、`<source_text>` 独立），不得在同一 `<qa_pair>` 重复 `<pic>`。**禁止**多图合并在同一 `<qa_pair>`；**禁止**无 `<pic>` 的 `<qa_pair>`（有图时）。若本段**完全无图**，才可输出纯文字 `<qa_pair>` 或 `<empty></empty>`。
4.1 **图题强绑定（强制）**：只要本段存在有效图片，所有 `<qa_pair>` 都必须围绕某个 `<pic>` 对应图生成；禁止输出与任何图片无关的纯文本泛化题（如未指向图号/图像的题目）。
4.2 **疾病篇问句封闭（强制）**：第二部分（疾病篇）同样只允许“题目白名单”句式；禁止生成任何“如何调护/食疗有哪些/如何鉴别/治疗原则”问句。
4.3 **治疗信息位置（强制）**：若原文包含治法方药，只能写入 `<solution>` 作为补充，不得作为 `<question>` 的提问目标。
5. 禁止输出任何与最终 XML 无关的前后缀文字（如“下面是结果”“说明如下”“继续”等）；输出首字符必须是 `<`，且应为 `<chapter>` 或 `<empty></empty>`。
6. `question` 文本中禁止“多图合问”模式：若出现“和/及/、”连接多个图号（如 `图X-X和图Y-Y`、`图X-X、Y-Y`），视为不合格，必须拆成多条单图 QA。

**第一部分（理论）图片**
- 解剖/分区图：每图 **1～5** 条 QA（证据不足可 1 条）；问句仅允许“结构/解剖图专用白名单”3句式；`<source_text>` 须含同号锚点原句。
- 舌象/舌苔图：每图 **1～5** 条（证据不足可 1 条）；问句仅允许“题目白名单”7句式；优先拆为“舌色题/舌形题/舌态题/苔质或苔色题/提示题”等细粒度单点题。

**第二部分（疾病）图片**
- 每图 **1～5** 条 QA（证据不足可 1 条）；问句只能从“题目白名单”中选取，不允许任何开放式问句。
- 该节若出现简易治疗/方药，可放入 `<solution>`，但不得出现在 `<question>`。
**证型与方剂匹配（重要提醒）**
- 若同一图片同时涉及证型判断与治疗方剂，请检查方剂与证型逻辑是否一致。
- 若不确定方剂是否匹配：可仅输出证型，把方剂留空；宁可留空，不要配错。

**禁止**
- 禁止把多张图合并进同一个 `<qa_pair>`。
- 禁止跳过本段 JSON 中应保留的教学图（仅装饰/页眉/纯重复可略）。
- 禁止编造不存在的 `img_path`。
- **问句禁冗余指图**：`<question>` 内禁止叠用「见图/参照图/结合图」等重复提示；一句一问，指图一次即可。
- **禁止多图合问**：禁止在同一 `<question>` 中出现两个及以上图号并列（如“图1-44、1-45、1-46分别…”）；必须拆分为 3 条单图 QA。
- **禁止多维合问**：禁止在同一 `<question>` 中并列两个及以上维度（如“舌质和舌苔…」「苔色和苔质…”）；必须拆成多条单维度题。
- **限制离图泛问**：禁止生成不含“图中”锚点、不锚定 `<pic>` 的泛问。
- **禁止脱图问句**：禁止出现不含“图中”锚点的问句（如“淡白湿润胖嫩舌有哪些伴随特征”）；必须改为“图中……”。
- **禁止占位式治疗答案**：禁止出现“暂无明确…/可参照…/可按常规调护…”等非抽取文本；治疗答案必须可由该图同号 `source_text` 逐字追溯。
- **禁止白名单外题干**：凡不属于“题目白名单”7种句式（或其“图中舌象”替代版）的一律判为不合格并重写。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文的目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出前自检（必须通过）：
1) 每条 `<question>` 是否只问一个维度？若含“和/及/分别”并跨维度，必须拆题。
2) 是否出现“舌质和舌苔”“苔质和苔色”同问？若是，判不合格并重写。
3) 是否完全未生成“如何调护/食疗有哪些/如何鉴别/治疗原则”这类白名单外问句？
4) `<answer>`/`<solution>` 是否含“暂无明确”“可参照”之类兜底语？若有，删除并按原文重写。
5) 每条 `<question>` 是否含“图中/图中的”锚点？若无，判不合格并重写。
6) `<question>` 是否仍含 `[` 或 `]` 占位符？若有，判不合格并重写（特别是“苔质/苔色”必须二选一）。
7) `<answer>` 是否与 `<question>` 同维度？若问单维度（舌色/舌形/舌态/舌苔），答案出现其它维度词则判不合格并重写。
8) `<source_text>` 是否含占位词（图注原文/第N条原文/原文/见原文/同上原文）？若有，判不合格并重写为「id:数字 + 原句」。
9) `<question>/<answer>/<source_text>` 的证据是否仅来自“显式同号句集合”？若出现不含同号图号的句子，判不合格并重写。
10) `<source_text>` 首条是否为同号锚点句（图M-N/（图M-N））或该图 `image_caption` 原文？若否，判不合格并重写或删条。
11) `<source_text>` 是否出现“若见/若…则/若…多提示”且该分句不含同号图号？若是，判不合格并重写。
12) 若存在“非图注显式同号句”，是否至少有 1 条 QA 已纳入该句（或与图注并列纳入）？若否，判不合格并重写。
13) 若同一原句含多个图号，是否已按当前 `figure_id` 做最小切片并保留当前图号？若未切片导致弃用可用证据，判不合格并重写。
14) 同图若已出现“是什么”与“提示了什么”，且同号证据中存在可抽取的形态特征短语，是否已补“有什么特征”题？若未补，判不合格并重写。
15) 若 `solution` 中出现“提示/主病/病机”而该图没有对应“提示了什么”问句，是否已补题？若未补，判不合格并重写。

输出结构示例（`<pic>` 路径须与输入 JSON 一致；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>（一）舌的形态特点</title>
<qa_pair><label>1</label><question>图中的舌形是什么？</question><answer>舌体偏胖嫩。</answer><solution>摘录该图图注与邻接说明。</solution><source_text>id:142 舌体胖嫩，边有齿痕；id:143 该象多见于脾虚湿盛。</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>图中展示的舌形有哪些特征？</question><answer>舌体较胖，边缘可见齿痕。</answer><solution>摘录该图对应特征描述。</solution><source_text>id:142 舌体较胖，边缘有齿痕；id:143 舌体松软，津液偏多。</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>3</label><question>图中的舌态提示了什么？</question><answer>提示气虚或阳虚倾向。</answer><solution>摘录该图同段可追溯病机句。</solution><source_text>id:143 同图病机原文一句</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>4</label><question>图中的舌象有什么特征？</question><answer>舌色淡，舌体偏胖，舌态稍缓，苔白润。</answer><solution>按图注顺序摘录多维特征，不扩写。</solution><source_text>id:144 图注与邻接 text 原文</source_text><pic>images/xxx2....jpg</pic></qa_pair>
</chapter>

<chapter><title>（一）病毒性肝炎</title>
<qa_pair><label>5</label><question>图中的舌色是什么？</question><answer>舌质偏红。</answer><solution>摘录该图图注句。</solution><source_text>id:681 舌质红，苔薄黄而少津，提示肺阴亏虚。</source_text><pic>images/yyy....jpg</pic></qa_pair>
<qa_pair><label>6</label><question>图中展示的舌态有哪些特征？</question><answer>可见少津表现。</answer><solution>摘录同图邻接说明，不引入它图内容。</solution><source_text>id:681 同图原文一句</source_text><pic>images/yyy....jpg</pic></qa_pair>
<qa_pair><label>7</label><question>图中的舌象提示了什么？</question><answer>提示肝肾阴虚（据原文）。</answer><solution>可补充本节与该图同段的治法方药原文（如有）。</solution><source_text>本节对应图注与邻接 text 照录</source_text><pic>images/yyy....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>请比较这几张图分别显示什么苔色？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
每张图各一条（每条各自一个 `<pic>`），且题干必须为白名单句式并含“图中”锚点。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<think>`/analysis/draft text.
"""
        return PROMPT

