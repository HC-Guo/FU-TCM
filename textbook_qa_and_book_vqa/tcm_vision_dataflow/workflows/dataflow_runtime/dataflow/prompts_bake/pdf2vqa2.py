from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

# 全书大致结构（许家佗《中医舌诊临床图解》），供模型对齐章节层级，勿凭此编造正文
_TONGUE_BOOK_OUTLINE = """
Book outline (use only to recognize hierarchy, not to invent text):
- 第一章 舌诊的基础知识
  - 什么是舌诊；舌诊的原理（组织结构、蕈状/丝状乳头、舌与脏腑经络气血津液、舌面脏腑分属）
  - 望舌的内容与方法（舌体、舌苔、舌下络脉；伸舌顺序、舌下络脉、刮舌验苔；注意事项）
  - 舌象的正常与异常、影响因素、如何掌握舌诊方法
- 第二章 中医舌诊的内容
  - 第一节 舌质：舌神（荣舌/枯舌）；舌色（淡红、淡白、枯白、红、绛、舌尖红、舌边尖红、青紫、淡紫、瘀斑、瘀点等）；舌形质（老嫩、胖大、肿胀、齿痕、瘦、红点、芒刺、裂纹、舌衄、舌疮等）；舌态（歪斜、僵硬、痿软、短缩、吐弄、震颤等）；舌下络脉
  - 第二节 舌苔：苔质（薄厚、润燥滑糙、腻苔各型、腐苔、脓腐、白霉、剥苔各型、地图舌、镜面舌等）；苔色（白苔各型、黄苔各型、灰黑苔、相兼苔色）
- 第三章 舌诊在临床病证诊断中的应用
  - 舌象的综合分析方法（神气胃气、舌体与舌苔综合、动态观察）
  - 第二节 临床舌象综合分析（各典型复合舌象条目，如淡紫舌、镜面红舌、剥苔等个案分析）
  - 第三节 常见基础证候的典型舌象特征（气虚、血虚、阴虚、阳虚、津液亏虚、气滞、血瘀、实寒、实热、痰湿等）
  - 第四节 舌象与病证的诊断治疗：单病单证举例（慢性胃炎各证、功能性消化不良、胆石症、糖尿病、冠心病、高血压、心律失常、肿瘤各证、呼吸系统与各杂病等）；「常见病证的诊断、治疗与舌象变化」各系统（脾胃、心脑血管、内分泌、妇科、肿瘤、呼吸、其他杂病）
- 第四章 体质与舌象
  - 体质概念与分类；九种典型体质的舌象及其干预（平和、气虚、阳虚、阴虚、痰湿、湿热、瘀血、气郁、特禀）
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
Document hierarchy hint (许家佗《中医舌诊临床图解》; do not invent text): Chapter 1 tongue-diagnosis basics; Chapter 2 tongue body and coating content; Chapter 3 clinical synthesis, pattern tongue signs, and disease-related diagnosis/treatment; Chapter 4 constitution and tongue. Prefer section titles as printed on the page.
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
你是许家佗《中医舌诊临床图解》的**VQA 数据生成**编辑。输入为**一个 JSON 数组**：每项有整数 **id**（按顺序编号）及正文字段；包含图片项（`img_path`、图注/说明文字等）。

**零容忍输出约束（优先级最高）**
- 你是**最终答案通道**，不是草稿通道。只允许输出最终 XML 结果。
- **严禁输出任何思考/草稿/解释文本**，包括但不限于：`<think>`、`<thinking>`、`<reasoning>`、`analysis`、`Chain-of-Thought`、`Now, ...`、`Let's ...`、"我先..." 等元叙述。
- 一旦输出任何上述内容，将导致下游解析失败并整批作废。请直接从 `<chapter>` 或 `<empty></empty>` 开始输出。

**范式转变（必须遵守）**
- **以图为主**：先枚举本段 JSON 中的每个 `img_path`，再围绕该图在顺序上**附近**找文字（图注、上下行说明、同小节标题），生成 QA。
- **文字为辅**：不要为了凑条数先写大量无图 QA；本段内**每张有效教学图须输出多条带该图 `<pic>` 的 `<qa_pair>`（见下条数）**。
- **`id` 递增 ≈ 阅读顺序**，就近配对图注与正文。
- **本书按章节定类（强制）**：从**第二章开始**生成图片 QA，**第一章图片一律跳过不生成**。
- **语言硬约束（强制）**：`<title>/<question>/<answer>/<solution>/<source_text>` 必须全部使用中文，不得输出英文问答或中英混排句。

{_TONGUE_BOOK_OUTLINE}

**核心任务：以图片为中心，按章节规则生成分类 QA（同一图可多条）。**

**章节-类别映射（强制）**
1. **第一章**：不生成 QA（即使有图，也跳过）。
2. **第二章**：每张有效图片优先生成以下两类（有证据才写）  
   - 舌象特征  
   - 临床意义
   - **覆盖约束（强制）**：第二章图谱范围必须覆盖到**图2-141**。若本段/本批输出中发现第二章图号未覆盖至图2-141（例如停在图2-124），不得提前结束；应继续在后续内容中补抽缺失图号，直至覆盖到图2-141。
3. **第三章**：按图号区间分三套规则（图号以图注或邻近正文可识别到的“图3-*”为准）  
   - **图3-1 ～ 图3-29**：舌象特征（舌质）、舌象特征（舌苔）、临床意义  
   - **图3-30 ～ 图3-77**：常见症状、舌象  
   - **图3-78 ～ 图3-209**：舌象特征、中医诊断、治则治法
4. **第四章**：每张有效舌象相关图片优先生成以下五类（有证据才写）  
   - 舌象特征  
   - 常见表现  
   - 发病倾向  
   - 心理特征  
   - 干预方案  
   **注意**：第四章若图片主要为穴位定位、经络取穴、按压操作等**非舌象主题**，该图**不生成 QA**。

**类别缺证据时的处理**
- 以上类别不是机械配额。若输入中缺少某类证据，可不生成该类。
- 但凡某类在图注/邻文中有明确依据，应尽量覆盖，不要遗漏。
- **最小产出判定（程序友好）**：
  - 对单张图，先遍历该章应生成的类别清单；
  - 每个类别若在 `image_caption`、邻近 `text`、`list_items` 中检索到直接证据，则该类别**必须生成 1 条** `<qa_pair>`；
  - 若检索不到直接证据，则该类别可跳过，不得臆造；
  - 每张应保留图片至少输出 1 条有证据 QA；若所有类别均无证据，则跳过该图。

**图片处理规则（最重要）**
1. **每张图片多条 `<qa_pair>`（强制）**：输入 JSON 中每个有 `img_path` 且非装饰/页眉类的图片项，按其章节类别生成**多条** `<qa_pair>`；同一图下不同类别分开成题。若该图邻接原文极少，至少保留 1 条有依据 QA。
2. **一图多题拆分**：同一图下可将证型结论、病机阐释、鉴别要点、治法方药/调护、临床意义等**拆入不同 `<qa_pair>`**；**禁止**把多张不同 `img_path` 合并进同一 `<qa_pair>`。
3. **图文配对优先级**（从高到低）：
   - 图注行（如「图1-1 舌背各部名称」「图2-1 …」）
   - 图片前最近的章节/小节标题
   - 图片后 1～3 行说明文字
   - 同章节内与图主题一致的「○…提示…」句或证候/病证说明句（第三章病证与舌象、第二章舌象图谱等）
4. **`<pic>`**：必须使用输入 JSON 中该项的 **`img_path` 原文**（与 JSON 一致），放在 `<qa_pair>` 内；**可**放在 `<question>` 末尾或 `</source_text>`/`</solution>` 之后，整个 `<qa_pair>` 内**恰好** `<pic>` 出现一次。
5. **问句单图约束（强制）**：`<question>` 只能询问当前这一张图，禁止出现多图并提语言。一条 `<qa_pair>` 只能对应一个 `img_path`。
6. **无图号问法（强制）**：问句默认使用“图中舌象/该图所示/据图中表现”等自然表述。
7. **问句禁图号（强制）**：`<question>` 中**禁止**出现任何图号模式，包括但不限于“图3-1”“图 3-1”“图3.1.1”“Fig.3-1”“3-1图示”等。
8. **问句禁专名（强制）**：`<question>` 中禁止直接写入具体病名、证型名、体质名或专有条目名（如“阳虚证”“血瘀证”“痰湿质”“慢性胃炎—脾胃气虚证”等）。统一改写为中性表达，如“图中舌象…/图中常见症状…/图中对应治则治法…”。

**防止误绑非舌象小图（强制）**
9. **先判图，再出题**：第三章生成 QA 前，先判断该 `img_path` 是否为“舌象主图”。若不是舌象主图，则该图不生成舌象类 QA。
10. **舌象主图判定依据（需至少满足一条）**：
   - 该图 `image_caption` 或邻近 `text` 明确描述舌体/舌质/舌苔/舌色/舌形/舌态等；
   - 该图在同段被用于“舌象特征/中医诊断/治则治法”的直接证据引用；
   - 该图对应图注为病例舌照，而非符号、流程、标记或说明性插图。
11. **必须排除的图片类型**：图标、箭头标记、角标、装饰性小插图、流程示意、局部符号说明图、纯文字框图、页眉页脚重复图。上述图片即使紧邻舌象正文，也不得当作 `<pic>` 绑定。
12. **同段多图防串图**：当同一段出现“舌象照片 + 旁侧小图标/示意图”时，只允许将 QA 绑定到舌象主图 `img_path`；旁侧图标不得生成该段舌象 QA。
13. **证据回溯校验**：每条 QA 的 `<source_text>` 必须能直接支撑该 `<pic>` 为舌象主图；若仅能支撑“说明图/标记图”，则该条 QA 作废并重选图片或跳过。
14. **宁缺毋错**：若无法确认某图是否为舌象主图，宁可跳过该图，不得冒险绑定。

**尺寸阈值硬过滤（强制，优先级高于语义；仅第三章/第四章启用）**
15. 若输入中可获得图片尺寸（如 `width`/`height`、bbox 宽高或等效像素信息），第三章与第四章必须先做尺寸过滤，再做语义判定。
16. **第二章不启用面积阈值过滤**：第二章图片不得因 `bbox_area` 或面积占比偏小而直接丢弃；仅按图注/邻文语义与图标类型规则排除明显非舌象图。
17. 第三章/第四章中，命中以下任一条件即判定为“图标/小插图候选”，**禁止**用于舌象 QA：
   - `min(width, height) < 150`；
   - `width * height < 100000`；
   - 同页存在更大图片时，当前图片面积 < 同页最大图片面积的 0.20。
18. 若同页同段有多张图，**允许存在多张舌象主图并分别生成 QA**。不要默认“只保留最大图”。
    - 仅当某图命中小图阈值（第17条）或明显属于图标/辅助示意时才排除；
    - 若同页有多张未命中小图阈值且图注/邻文均支持舌象解读的图片，均应保留为独立候选图。
19. 第三章/第四章中，尺寸与语义冲突时，以尺寸过滤优先：即使邻文提到该图，若尺寸命中小图阈值，也不得绑定为 `<pic>`。
20. 若当前输入完全不含尺寸信息，退回执行第 9～14 条语义规则；不得因缺少尺寸信息而放宽“小图误绑”限制。

**舌象措辞：图注优先（强制）**
- 图注行通常最可靠（如「图2-1 舌质红而少津：提示肝肾阴虚」「图1-5 淡红舌：正常舌色」）。**问句里的舌象描述应优先逐字或紧缩摘自该图对应图注**，再与「○…提示…」句核对；二者冲突时**以图注为准**。
- 若无独立图注但有「（图x-x）」夹在正文内，取该括号前后最短完整的舌象短语写入问句。
- 禁止把**其它条目**（邻近段落、同页它病）的舌象词贴到本图上。

**图文一致性（强制）**
- 本流水线以 JSON 文本输入为主、模型未必直接看到像素时：以该图对应 **图注 + 紧邻说明** 作为「肉眼所见」的**权威文字代理**，问句舌象须与之对齐，不得矛盾。
- `<question>` 所写舌象（**舌质/舌苔/津液/舌形/舌态** 等）**必须与**该 `<pic>` 在书中所附**图注及图意**一致，不得与图意矛盾。
- **禁止**仅依据离图很远的正文「想当然」改写字样：若图注写「舌质红而少津」，问句**不得**写成「苔黄」；若图注未出现「苔黄」，**不得**擅自加入。
- 若无法从图注/紧邻说明确定具体舌象用语：问句用 **「图中舌象」** 指代（可带疾病名与图号），**不要编造**细到苔色、津液等细节。
- `<answer>` 的证型/病机须与同图图注「提示……」及原文一致；`<solution>` 只收录**该书该图/该病节**下的治法方药，勿串页。

**问句多样化（示例引导，非硬模板）**
- 同一图的多条题目需围绕不同类别分工，避免重复句式。
- 建议使用无图号表达：如“图中舌象有何特征？”“该图所示舌象提示什么？”“根据图中表现可归纳怎样的中医诊断？”“该图对应治则治法如何概括？”“该体质图可见哪些常见表现/发病倾向/心理特征/干预方案？”。
- 禁止机械套模板；应结合该图图注与邻文自然改写。

**问句模板示例（简洁版，非强制）**
- 以下每类给出 1～2 个示例模板，仅用于稳定语义边界；生成时可自由改写，不必逐字照抄。
- **非强制**：不要求固定词面、不要求固定句式；但该条问句表达的类别语义必须与对应类别一致。
- 第二章
  - 舌象特征：如「图中舌象有什么特点？」「请概括该图舌质舌苔表现。」
  - 临床意义：如「该图舌象提示什么？」「这个舌象在临床上有什么意义？」
- 第三章 图3-1～图3-29
  - 舌质特征：如「该图舌质表现如何？」「请描述图中舌质特点。」
  - 舌苔特征：如「该图舌苔表现如何？」「请概括图中苔质苔色情况。」
  - 临床意义：如「该图舌象的临床提示是什么？」「从该图可判断哪些临床意义？」
- 第三章 图3-30～图3-77
  - 常见症状：如「该图对应常见症状有哪些？」「结合该图可见哪些症状表现？」
  - 舌象：如「该图舌象应如何概括？」「图中舌象属于什么表现？」
- 第三章 图3-78～图3-209
  - 舌象特征：如「该图舌象特征是什么？」「请概括图中舌质舌苔特点。」
  - 中医诊断：如「根据该图应如何做中医诊断？」「该图更支持哪类辨证判断？」
  - 治则治法：如「针对该图表现应采用什么治则治法？」「该图对应的治法如何把握？」
- 第四章
  - 舌象特征：如「图中舌象特征是什么？」「请概括图中舌象表现。」
  - 常见表现：如「图中常见表现有哪些？」「从该图可归纳哪些常见表现？」
  - 发病倾向：如「图中对应的发病倾向是什么？」「该图提示的易感方向是什么？」
  - 心理特征：如「图中心理特征是什么？」「从该图可概括哪些心理特点？」
  - 干预方案：如「图中对应的干预方案有哪些？」「该图的调理干预应如何进行？」

**问答生成（字段分工）**
- `<question>`：遵守上文**图注优先**、**图文一致**、**单图单问**与**无图号问法**。问句要能体现当前类别（如舌象特征/临床意义/常见症状/中医诊断/治则治法/常见表现/发病倾向/心理特征/干预方案）。
- `<answer>`：**短答**——先给出与图注一致的舌象/结构结论或证型名；**并尽量**用**自己的话**压缩，且**至少融入一条**来自图注**以外**的信息：同一 JSON 块内该图**前后文**、或**相邻 id** 中与该图主题直接相关的病机句、鉴别句、临床意义句（若输入里确有）。**禁止** `<answer>` 仅为图注的逐字缩写或只加「（据图注）」而无新增信息——若输入除图注外无可融合句，也须**改写句式**并点明「见于教材该图」类教学化表述，避免与图注完全同形。
- `<solution>`：**学习性长段（与 `<answer>` 分工）**——从输入 JSON 中**检索并摘录**（允许适度紧缩，勿改写证型/方名核心字）：
  - **第二章**：优先补充舌象解释与临床意义；
  - **第三章**：按图号区间补充常见症状/中医诊断/治则治法等；
  - **第四章**：补充常见表现、发病倾向、心理特征、干预方案等；
  - 若该类无证据可留空，禁止编造。
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
4. **一图多题 + `<pic>` + `<source_text>`（强制）**：本段 JSON 中每个应保留的有效 `img_path`（非装饰/页眉，且满足章节规则）需输出对应类别的多条 `<qa_pair>`；每条 `<qa_pair>` 内**恰好一个** `<pic>`，且**必须**含 `<source_text>`。**禁止**多图合并在同一 `<qa_pair>`；**禁止**无 `<pic>` 的 `<qa_pair>`（有图时）。
5. 禁止输出任何与最终 XML 无关的前后缀文字（如“下面是结果”“说明如下”“继续”等）；输出首字符必须是 `<`，且应为 `<chapter>` 或 `<empty></empty>`。
6. `question` 文本中禁止“多图合问”模式：若出现“和/及/、”连接多个图片对象，视为不合格，必须拆成多条单图 QA。
7. `question` 文本中禁止出现图号或其变体（连字符、点分、英文 Fig 前缀等）；违反则整条视为不合格并重写。
8. `question` 文本中禁止出现具体病名/证型名/体质名等专名；需改为“图中…”“该图所示…”等中性问法（例如“阳虚证舌象有何特征？”改为“图中舌象有何特征？”）。
9. 第二章不得在图号覆盖未达图2-141时停止抽取；若当前批次未达图2-141，需继续补齐缺失图号对应的有效舌象图 QA。

**章节执行细则（再次强调）**
- 第一章：跳过，不建 QA。
- 第二章：仅围绕“舌象特征、临床意义”建题。
- 第三章：
  - 图3-1～图3-29：舌质特征、舌苔特征、临床意义。
  - 图3-30～图3-77：常见症状、舌象。
  - 图3-78～图3-209：舌象特征、中医诊断、治则治法。
- 第四章：舌象特征、常见表现、发病倾向、心理特征、干预方案；穴位相关非舌象图跳过。

**证型与方剂匹配（重要提醒）**
- 若同一图片同时涉及证型判断与治疗方剂，请检查方剂与证型逻辑是否一致。
- 若不确定方剂是否匹配：可仅输出证型，把方剂留空；宁可留空，不要配错。

**禁止**
- 禁止把多张图合并进同一个 `<qa_pair>`。
- 禁止跳过本段 JSON 中应保留的教学图（仅装饰/页眉/纯重复可略）。
- 禁止编造不存在的 `img_path`。
- 禁止在问句中强制堆叠图号、页码或“见图”尾注；保持自然单图提问。
- 禁止多图合问：同一 `<question>` 不得并列多个图片对象。

**必须丢弃的正文（不建 QA）**
- 版权、出版信息、纯页码、重复页眉、无正文的目录行。

若本段 JSON 无可抽取正文：只输出 `<empty></empty>`
"""
        PROMPT += """
输出结构示例（`<pic>` 路径须与输入 JSON 一致；**一图须多条 `<qa_pair>`**；每条含 `<source_text>`；紧挨标签、少换行）：

<chapter><title>一、舌的组织结构</title>
<qa_pair><label>1</label><question>图中舌象有何特征？</question><answer>舌质淡红，舌体柔软，苔薄白。</answer><solution>可据邻文补充观察要点与判读注意事项。</solution><source_text>id:142 图注原文；id:143 邻接 text 原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
<qa_pair><label>2</label><question>该图所示舌象有何临床意义？</question><answer>提示气血津液相对协调，偏向常态或病势较轻。</answer><solution>结合同段原文说明其在辨证中的提示价值。</solution><source_text>id:143 邻接 text 原文</source_text><pic>images/xxx....jpg</pic></qa_pair>
</chapter>

<chapter><title>慢性胃炎—脾胃气虚证</title>
<qa_pair><label>3</label><question>图中舌象对应的中医诊断如何概括？</question><answer>可归纳为脾胃气虚证，舌淡胖并见齿痕。</answer><solution>诊断依据与证候要点据本节原文摘录。</solution><source_text>id:201 证候描述原文；id:202 图注原文</source_text><pic>images/yyy....jpg</pic></qa_pair>
<qa_pair><label>4</label><question>针对该图所示证候，治则治法应如何把握？</question><answer>以健脾益气、和胃化湿为主。</answer><solution>补入本节对应方药或调护条文；无证据则留空。</solution><source_text>id:203 治法方药原文</source_text><pic>images/yyy....jpg</pic></qa_pair>
</chapter>

错误示例（禁止）：
<qa_pair><label>9</label><question>图A和图B分别提示什么？</question>...</qa_pair>
上例为一问多图，必须拆成三条：
图A 一条、图B 一条、图C 一条（每条各自一个 `<pic>`）。

Now process the provided json and output final result only.
Reminder: output must begin with `<chapter>` or `<empty></empty>`; never output `<think>`/analysis/draft text.
"""
        return PROMPT

