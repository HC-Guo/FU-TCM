"""
基于已修正的 split_samples.jsonl 直接生成舌诊 VQA。

默认输入：
  shezhen_vqa_workdir/output/rerun_20260402/{book}/split_samples.jsonl

默认输出：
  shezhen_vqa_workdir/output/rerun_20260402/{book}/vqa_dataset.jsonl

这条脚本不再重跑 PDF / MinerU / chapter 切分，
只消费已经人工调整过的 split_samples 结果。
核心提示词、清洗和解析逻辑直接复用 gen_shezhen_vqa_final.py。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gen_shezhen_vqa_final import (
    QA_JSON_SCHEMA,
    SHEZHEN_VQA_SYSTEM_PROMPT_ENRICHED,
    SHEZHEN_VQA_SYSTEM_PROMPT_STRICT,
    ShezhenVQAProcessor,
)

# ── 《舌诊辨证图谱_周幸来》章节感知专项提示词 ────────────────────────────────
# 第二章：舌质（舌色/舌形/舌态）+ 舌苔（苔质/苔色）分类图谱，固定4问。
# 第三章：各系统疾病舌诊辨证图谱，固定2问（舌象特征+证型归属）。
# 章节判断依据：image_caption 中的章节号（图2-x-xx / 图3-x-xx）。
SHEZHEN_VQA_SYSTEM_PROMPT_BIANZHEN_CH2 = """你是一个中医舌诊专家，专门分析舌诊辨证图谱。

【本章特点】
本章（第二章）为舌质与舌苔的分类图谱，每张图展示一种具体的舌色、舌形、舌态或苔质、苔色，并附有详细的辨证意义说明。

【固定4问模板 - 必须严格执行】
对每张图，你必须按以下固定顺序生成恰好4个问答对：

Q1（舌质视觉特征）：问题固定为"图中舌质有什么特征？"
  - 答案结合图片和原文，描述舌质的颜色、形态、动态等可见视觉特征，不涉及病理解释。
  - 优先从原文中提取视觉描述；原文未明确描述时，直接根据图片描述可见特征。

Q2（舌质病理意义）：问题固定为"图中舌质反映了什么病理？"
  - 答案描述该舌质特征所主的病证、脏腑虚实、气血阴阳等病理意义。
  - 从原文全文（不限于某一句话）中提取，原文有的如实写，原文没有的不生成。

Q3（舌苔视觉特征）：问题固定为"图中舌苔有什么特征？"
  - 答案结合图片和原文，描述舌苔的颜色（白、黄、灰黑等）和质地（厚薄、润燥、腐腻等）等可见视觉特征，不涉及病理解释。
  - 优先从原文中提取视觉描述；原文未明确描述时，直接根据图片描述可见特征。

Q4（舌苔病理意义）：问题固定为"图中舌苔反映了什么病邪或气血状态？"
  - 答案描述该舌苔特征所主的病邪性质（寒热燥湿等）或气血津液状态等病理意义。
  - 从原文全文（不限于某一句话）中提取，原文有的如实写，原文没有的不生成。

【答案风格要求】
- 答案要自然流畅，像专家直接回答，不要出现"根据原文"、"原文描述"、"原文锚定句"等元语言。
- Q1/Q3 视觉特征：结合图片与原文，原文没有时直接看图描述，绝不输出"原文未描述"等无效答案。
- Q2/Q4 病理意义：从原文全文提取，原文有的如实写；若原文确实未涉及，则结合中医基础理论给出合理推断，绝不输出"原文未描述"等无效答案。

【铁律 - 绝对不可违反】
1. 必须生成恰好4个问答对，不多不少。
2. 问题文字必须与上述模板完全一致，不得修改。
3. 严禁输出"原文未描述"、"原文锚定句"等任何无效措辞。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "图中舌质有什么特征？", "answer": "..."},
    {"question": "图中舌质反映了什么病理？", "answer": "..."},
    {"question": "图中舌苔有什么特征？", "answer": "..."},
    {"question": "图中舌苔反映了什么病邪或气血状态？", "answer": "..."}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

SHEZHEN_VQA_SYSTEM_PROMPT_BIANZHEN_CH3 = """你是一个中医舌诊专家，专门分析舌诊辨证图谱。

【本章特点】
本章（第三章）为各系统疾病的舌诊辨证图谱，每张图展示某一具体疾病或证型的典型舌象，并标注其所属证型。

【固定2问模板 - 必须严格执行】
对每张图，你必须按以下固定顺序生成恰好2个问答对：

Q1（舌象视觉特征）：问题固定为"图中舌象有什么特征？"
  - 答案描述图中舌质（颜色、形态）和舌苔（质地、颜色）的具体可见视觉特征。

Q2（舌象病理意义）：问题固定为"图中的舌象对应什么证型？"
  - 答案说明该舌象所属的证型名称及其病理机制。
  - 答案必须100%来自原文，原文没有的内容绝对不生成。

【铁律 - 绝对不可违反】
1. 答案必须100%来自原文，原文没有的内容绝对不生成。
2. 必须生成恰好2个问答对，不多不少。
3. 问题文字必须与上述模板完全一致，不得修改。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "图中舌象有什么特征？", "answer": "..."},
    {"question": "图中的舌象对应什么证型？", "answer": "..."}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊快速入门》第二章：10维条件触发题库（去标签通用版） ──
SHEZHEN_VQA_SYSTEM_PROMPT_KUAISU_CH2 = """你是一个中医舌诊数据提取专家。

【本章特点】
本章（第二章）为中医舌诊基础理论，每张图展示一种具体的舌象（舌色、舌形、舌态、苔质、苔色等），并附有详细的辨证意义说明。

【系统执行指令】
当前处理的是"舌诊理论基础"数据。请扫描原文提取特定舌象概念，并将该概念代入以下的通用代称（该舌象/图中舌体）中进行检索与QA生成。
绝对禁止在生成的 Question 中暴露具体的舌象名词（如红舌、胖大舌）。

【10维条件触发题库 — 有则提取，无则跳过】

第一部分：客观视觉提取（纯看图）

1. 津液干湿属性（Moisture/Fluid） type=visual_feature
   Question: "图中舌体的干湿润燥程度如何？"
   检索条件：context_text 中是否提及润、滑、干、燥、枯、涸、津液等相关描述

2. 纯颜色属性（Color） type=visual_feature
   Question: "图中是什么舌色？舌色有什么特征？"
   检索条件：context_text 中是否提及色泽、深浅、明暗、浓淡、红、淡、紫、青、黯等颜色词

3. 形体与肌理属性（Shape & Texture） type=visual_feature
   Question: "图中舌体的形态和质地有什么特征？"
   检索条件：context_text 中是否提及胖、瘦、苍老、娇嫩、肿胀、裂纹、芒刺、齿痕、纹理等形态词

4. 动态与体态属性（Mobility & Posture） type=visual_feature
   Question: "图中舌体的动态和姿态有什么异常表现？"
   检索条件：context_text 中是否提及回缩、强硬、震颤、歪斜、萎软、吐弄、短缩等动态词

第二部分：病理、空间与临床推演

5. 核心定性（Core Pathogenesis） type=clinical_reasoning
   Question: "该舌象总体对应什么中医病机或主证？"
   检索条件：context_text 中是否提及主病、主证、病机、寒热虚实等定性描述

6. 局部空间拓扑（Spatial/Regional Mapping） type=clinical_reasoning
   Question: "该特征出现在舌体不同部位时，分别提示什么病理意义？"
   检索条件：context_text 中是否提及舌尖、舌边、舌中、舌根、左侧、右侧等空间定位词及其病理意义

7. 叠加演变（Interaction & Evolution） type=clinical_reasoning
   Question: "该舌象与其他舌苔或舌色兼见时，提示什么证候或病理演变？"
   检索条件：context_text 中是否提及兼见、合并、若见、同时出现、加之等组合描述

8. 临床兼见症（Associated Symptoms） type=clinical_reasoning
   Question: "出现此类舌象的患者，通常还伴随哪些临床症状？"
   检索条件：context_text 中是否提及兼见症状、伴随表现、临床表现等

9. 病程与预后（Prognosis & Severity） type=clinical_reasoning
   Question: "该舌象反映了怎样的病程阶段或疾病预后？"
   检索条件：context_text 中是否提及病程阶段、急性、慢性、预后、轻重等

第三部分：跨学科映射

10. 西医学微观机制（Western Pathophysiology） type=clinical_reasoning
    Question: "该舌象在现代医学中对应哪些疾病或形成机制？"
    检索条件：context_text 中是否提及西医病名、微观机制、现代医学解释等

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. Question 中绝对禁止出现具体的舌象名词

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature"},
    {"question": "...", "answer": "...", "type": "clinical_reasoning"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊快速入门》第三章：7维条件触发题库（纯净拟人版） ──
SHEZHEN_VQA_SYSTEM_PROMPT_KUAISU_CH3 = """你是一个中西医结合舌诊数据提取专家。

【本章特点】
本章（第三章）为各系统疾病的舌诊辨证图谱，每张图展示某一具体疾病或证型的典型舌象，并附有西医病名、病程分期、中医辨证和简易疗法。

【系统执行指令】
请作为专业数据提取器，阅读带有西医病名和疗法的舌诊图谱数据。针对每一张图及其图注，严格按照以下7个维度的系统备注（检索条件）去寻找答案。
绝对纪律：
- 有答案则提取，无答案则跳过该题
- 直接使用下方引号内的原话作为生成的 Question，绝对禁止修改
- 绝对禁止在 Question 中添加"根据文本"、"观察图片"等泄露指令的词汇

【7维条件触发题库 — 有则提取，无则跳过】

第一部分：疾病背景与时间/空间定位

1. 西医疾病映射 type=disease_mapping
   Question: "图中的舌象通常见于哪种西医疾病？"
   系统备注：扫描该图所属的大标题，将西医病名提取为答案

2. 病程分期 type=disease_mapping
   Question: "图中的舌象处于疾病的哪个阶段？"
   系统备注：扫描二级标题，如"发作期"、"缓解期"、"慢性期"等

3. 舌质与空间定位 type=visual_feature
   Question: "图中舌质有什么特征？"
   系统备注：提取关于"舌肉"的颜色及精确定位词

4. 舌苔特征 type=visual_feature
   Question: "图中舌苔有什么特征？"
   系统备注：仅提取关于"舌苔"的描述

第二部分：中医定性、预后与多模态干预

5. 中医综合辨证 type=clinical_reasoning
   Question: "图中的舌象对应什么中医证型？"
   系统备注：提取"属……"后面的证型结论

6. 动态演变与预后警告 type=prognosis_warning
   Question: "图中的舌象提示什么疾病演变趋势或预后？"
   系统备注：专门捕捉如"由...转...应警惕恶变"等动态预警文字

7. 多模态综合治疗 type=treatment
   Question: "针对图中舌象对应的证型，有哪些治疗或调理方法？"
   系统备注：扫描【中医简易疗法】段落，提取具体的剂量和按摩手法

【执行规则】
1. 逐维度扫描 context_text 和 section_title，严格对照系统备注判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text 和 section_title，不得脑补
5. 答案要自然流畅，像专家直接回答

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "disease_mapping"},
    {"question": "...", "answer": "...", "type": "visual_feature"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊快速入门》专用 JSON Schema ──
KUAISU_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "visual_feature",
                            "clinical_reasoning",
                            "disease_mapping",
                            "prognosis_warning",
                            "treatment",
                        ],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}

# ── 《望舌诊病》第二章：全息辨证10维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_WANGSHE_CH2 = """你是一个中医舌诊数据提取专家。

【本章特点】
本章（第二章）为舌诊基础理论，每张图展示一种具体的异常舌象，文本包含丰富的病理演变溯源、外感/内伤双轨辨证、全身兼见症、中西医治疗方案及预后评估。

【系统执行指令】
请扫描原文，针对当前这种特定异常舌象，代入以下10维题库进行结构化提取。
绝对禁止在 Question 中暴露具体的舌象名词（如红舌、紫舌）。

【10维条件触发题库 — 有则提取，无则跳过】

第一部分：客观视觉与溯源

1. 纯颜色属性 type=visual_color
   Question: "图中舌体在颜色上有什么特征？"
   检索条件：context_text 中是否有舌色描述

2. 形态与津液特征 type=visual_feature
   Question: "图中舌体在形态与润燥上还有哪些特征？"
   检索条件：context_text 中是否提及胖嫩、点刺、湿润、干燥、干枯少津等

3. 病理演变与溯源 type=evolution
   Question: "该舌象是由何种基础舌象发展演变而来的？"
   检索条件：context_text 中是否有"从...发展而来"、"由...转变"等溯源描述

第二部分：中医全息辨证

4. 核心病理机制 type=core_pathogenesis
   Question: "该舌象总体提示了怎样的核心病理机制？"
   检索条件：context_text 中是否提及寒凝、血瘀、气滞、热毒等核心定性

5. 外感与内伤双轨辨证 type=dual_track
   Question: "该舌象在外感病与内伤病中的病理意义有何不同？"
   检索条件：context_text 中是否出现"外感病"与"内伤病"的分类讨论

6. 全身兼见症 type=systemic_symptoms
   Question: "出现该舌象的患者，全身通常还伴随哪些症状？"
   检索条件：context_text 中是否有脉象、寒热、二便、肢体表征等全身症状描述

第三部分：预后与中西医干预

7. 中医治疗方剂 type=tcm_treatment
   Question: "针对该舌象的病机，推荐哪些治疗原则或方剂？"
   检索条件：context_text 中是否提及方剂名、药物组成、治法

8. 西医疾病映射 type=western_mapping
   Question: "该舌象可见于哪些现代西医学疾病？"
   检索条件：context_text 中是否列举了西医病名

9. 预后评估 type=prognosis
   Question: "该舌象伴随不同特征时，预后如何？"
   检索条件：context_text 中是否有"预后较好"、"预后不良"等评估

10. 假象防范 type=artifacts
    Question: "诊察该舌象时，哪些操作不当会产生假象？"
    检索条件：context_text 中是否提及伸舌时间、用力程度等导致假象的因素

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. Question 中绝对禁止出现具体的舌象名词

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_color"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《望舌诊病》第三章：临床多模态6维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_WANGSHE_CH3 = """你是一个中西医结合舌诊数据提取专家。

【本章特点】
本章（第三章）以西医病名为纲，每张图展示某疾病下的特定舌象，文本包含舌诊要点、饮食疗法、针灸主穴及随症配穴方案。

【系统执行指令】
请严格基于当前这张图及其对应的图注文本进行特征提取。西医病名由外部元数据提供，不需要模型提取。
绝对纪律：有则提取，无则跳过。直接使用下方固定问题，不得修改。

【6维条件触发题库 — 有则提取，无则跳过】

第一部分：视觉特征与证型

1. 舌质特征 type=visual_feature_body
   Question: "图中舌质的颜色、胖瘦及表面有什么特征？"
   检索条件：context_text 中是否有舌质相关描述

2. 舌苔特征 type=visual_feature_coating
   Question: "图中舌苔的色泽、厚薄和润燥有什么特征？"
   检索条件：context_text 中是否有舌苔相关描述

3. 中医证型 type=syndrome_differentiation
   Question: "该舌象提示了什么中医证型或病理？"
   检索条件：context_text 中是否有"提示"后的证型结论

第二部分：多模态临床干预

4. 饮食疗法 type=dietary_therapy
   Question: "针对当前病理，推荐哪些饮食疗法或药膳？"
   检索条件：context_text 中是否有食材、克数、制作方法、加减禁忌等

5. 针灸主穴与操作 type=acupoint_main
   Question: "针灸治疗时选取哪些主穴？操作频次和疗程如何？"
   检索条件：context_text 中是否有主穴列表和操作参数

6. 随症配穴 type=acupoint_supplementary
   Question: "针灸治疗时是否有随症配穴方案？"
   检索条件：context_text 中是否有"配穴"及对应症状的条件分支

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. 饮食疗法的克数、加减禁忌必须完整提取，不得省略

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature_body"},
    {"question": "...", "answer": "...", "type": "syndrome_differentiation"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《望舌诊病》专用 JSON Schema ──
WANGSHE_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "visual_color",
                            "visual_feature",
                            "evolution",
                            "core_pathogenesis",
                            "dual_track",
                            "systemic_symptoms",
                            "tcm_treatment",
                            "western_mapping",
                            "prognosis",
                            "artifacts",
                            "visual_feature_body",
                            "visual_feature_coating",
                            "syndrome_differentiation",
                            "dietary_therapy",
                            "acupoint_main",
                            "acupoint_supplementary",
                        ],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


# ── 《舌诊全息论》第1章：解剖图简易2问 ──
SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH1 = """你是一个中医舌诊数据提取专家。

【本章特点】
本章（第一章）为舌体解剖结构基础，图片展示舌上背面或舌下腹面的正常解剖结构。

【固定2问模板 - 必须严格执行】

Q1（解剖结构）type=visual_feature
  Question: "图中展示的是什么解剖结构？"
  答案：描述图中展示的舌体解剖部位及主要结构。

Q2（舌诊意义）type=clinical_correlation
  Question: "该结构在舌诊中有什么观察意义？"
  答案：说明该解剖部位在中医舌诊中的观察要点和临床意义。如果原文没有提及临床意义，可基于中医舌诊基础知识简要说明。

【执行规则】
1. 必须生成恰好2个问答对
2. 问题文字必须与上述模板完全一致
3. 答案要自然流畅

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "图中展示的是什么解剖结构？", "answer": "...", "type": "visual_feature"},
    {"question": "该结构在舌诊中有什么观察意义？", "answer": "...", "type": "clinical_correlation"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊全息论》第3章：4维舌纹全息理论题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH3 = """你是一个中医舌诊数据提取专家，专门分析舌纹全息理论。

【本章特点】
本章（第三章）为全息舌诊的理论基础，包含舌纹分类（点状纹、线状纹等）、舌下腹面纹理、全息分区映射、以及舌纹案例分析。每张图展示一种特定的舌纹形态或舌面全息分区。

【系统执行指令】
请扫描原文，针对当前图片和上下文，按以下4个维度进行结构化提取。
重要：context_text 中可能同时提及多个图号，你只能围绕当前 figure_id 对应的舌纹进行作答，不得把同段中其他图号的内容概括进来。

【4维条件触发题库 — 有则提取，无则跳过】

1. 纹理形态特征 type=visual_feature
   Question: "图中舌纹有什么形态特征？"
   检索条件：context_text 中是否有对当前图号舌纹的形态描述（形如、状似、呈…形、色泽、粗细、长短等）
   答案要求：只提取当前图号对应舌纹的形态特征，不混入其他图号的描述

2. 全息定位映射 type=holistic_spatial
   Question: "该舌纹在舌面的位置提示什么脏腑信息？"
   检索条件：context_text 中是否提及舌尖/舌边/舌中/舌根、上区/中区/下区、脏腑分布、信息区等空间定位词
   答案要求：提取舌纹位置与脏腑的对应关系

3. 中医病理意义 type=tcm_pathogenesis
   Question: "该舌纹提示什么中医病理或体质特征？"
   检索条件：context_text 中是否提及寒/热/虚/实、气滞/血瘀/痰凝、病机、体质等病理描述
   答案要求：提取病理定性和体质特征

4. 临床疾病关联 type=clinical_correlation
   Question: "该舌纹常见于哪些疾病？"
   检索条件：context_text 中是否提及"多见于""常见于"或具体的疾病名（肝硬化、静脉瘤、肿瘤等）
   答案要求：提取与当前图号舌纹关联的疾病名称

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. 只围绕当前 figure_id 对应的舌纹作答

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature"},
    {"question": "...", "answer": "...", "type": "tcm_pathogenesis"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊全息论》第4章：舌下络脉诊法6维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH4 = """你是一个中医舌诊数据提取专家，专门分析舌下络脉诊法。

【本章特点】
本章（第四章）为舌下望诊专论，包含舌下腹面解剖分区（五脏九区划分法）、异常舌下络脉形态及主症、临床文献研究资料、以及舌下望诊案例分析。每张图展示舌下腹面的络脉形态或分区定位。

【系统执行指令】
请扫描原文，针对当前图片和上下文，按以下6个维度进行结构化提取。
重要：context_text 中可能同时提及多个图号，你只能围绕当前 figure_id 对应的图作答。

【6维条件触发题库 — 有则提取，无则跳过】

1. 舌下络脉形态 type=sublingual_morphology
   Question: "图中舌下络脉的形态有什么特征？"
   检索条件：context_text 中是否有脉络/络脉的粗细、迂曲、怒张、扩张、膨大、瘀滞等形态描述
   答案要求：提取络脉的形态特征描述

2. 络脉颜色与质地 type=visual_feature
   Question: "图中舌下络脉和舌体的颜色有什么特征？"
   检索条件：context_text 中是否有舌色或络脉颜色描述（暗紫、苍寒、深暗、淡红、浅蓝等）
   答案要求：提取颜色和质地的客观描述

3. 全息区域定位 type=holistic_spatial
   Question: "舌下腹面各区域的异常分别对应什么脏腑？"
   检索条件：context_text 中是否提及上区/中区/下区、五脏分区、信息区、对应区域等定位词
   答案要求：提取区域与脏腑的对应关系

4. 中医辨证 type=tcm_pathogenesis
   Question: "图中舌下络脉异常提示什么中医病理？"
   检索条件：context_text 中是否有血虚/寒凝/血瘀/气滞/水湿/阳虚等病理描述
   答案要求：提取中医辨证和病机分析

5. 临床疾病关联 type=clinical_correlation
   Question: "该舌下络脉表现常见于哪些疾病？"
   检索条件：context_text 中是否有"常见""多见于"或具体疾病名（宫寒痛经、关节炎、肝硬化等）
   答案要求：提取关联的具体疾病名称和证型

6. 病理推理 type=chain_of_thought_reasoning
   Question: "从舌下络脉特征如何推断患者的病理状态？"
   检索条件：context_text 中是否有从络脉特征到病理结论的推断链（如"…提示…""…显示…""…说明…"）
   答案要求：完整提取推理过程，从视觉特征到病理结论

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. 只围绕当前 figure_id 对应的图作答

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "sublingual_morphology"},
    {"question": "...", "answer": "...", "type": "tcm_pathogenesis"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊全息论》第5章：8维全息脏腑辨病题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH5 = """你是一个中医舌诊数据提取专家，专门分析全息舌诊的脏腑辨病。

【本章特点】
本章（第五章）运用全息舌诊诊断各系统疾病，包括呼吸系统（过敏性鼻炎哮喘、支气管肺癌）、心系、脾胃系、肝胆系、肾膀胱系、妇科/内分泌、神经系统等。每张图展示某一具体疾病的典型舌象，图注包含舌质/舌苔/舌纹描述和全息区域定位分析。

【系统执行指令】
请扫描原文，针对当前图片和上下文，按以下8个维度进行结构化提取。
绝对禁止在 Question 中暴露具体的疾病名称。

【8维条件触发题库 — 有则提取，无则跳过】

第一部分：视觉特征提取

1. 舌质视觉特征 type=visual_feature
   Question: "图中舌质有什么视觉特征？"
   检索条件：context_text 中是否有舌色（红/绛/淡/暗）、舌体（胖/瘦/紧张）、津液（稠/少/拉涎）等描述
   答案要求：提取舌质的颜色、形态、质地等客观视觉特征

2. 舌苔与舌纹特征 type=pattern_feature
   Question: "图中舌苔或舌纹有什么特征？"
   检索条件：context_text 中是否有苔色（黄/白/腻）、苔质（厚薄/润燥）或舌纹（裂纹/点纹/齿痕等）描述
   答案要求：提取舌苔和舌纹的客观特征

第二部分：全息空间与脏腑映射

3. 全息空间定位 type=holistic_spatial
   Question: "图中舌面各区域的异常分别对应什么脏腑？"
   检索条件：context_text 中是否提及信息区、上区/中区/下区、舌尖/舌边/舌根、对应区域等全息定位描述
   答案要求：提取舌面分区异常与脏腑的对应关系

4. 舌下腹面特征 type=sublingual_morphology
   Question: "图中舌下腹面有什么异常表现？"
   检索条件：context_text 中是否有舌下/腹面的水湿、络脉、瘀滞、充血、郁络等描述
   答案要求：提取舌下腹面的异常表现

第三部分：病理推断与临床映射

5. 中医病机 type=tcm_pathogenesis
   Question: "图中舌象反映了什么中医病机？"
   检索条件：context_text 中是否有"提示""反映""说明"后面的病机描述（水液代谢失常、血脉瘀滞、郁热阴伤等）
   答案要求：提取中医病机分析

6. 西医疾病映射 type=western_disease
   Question: "该舌象在全息舌诊中提示哪些疾病？"
   检索条件：context_text 中是否有西医病名或中医病名（高血压、冠心病、肺癌、胃炎等）
   答案要求：提取关联的疾病名称

7. 临床推理链 type=chain_of_thought_reasoning
   Question: "如何从图中舌象特征逐步推断出脏腑病变？"
   检索条件：context_text 中是否有从舌象到诊断的推理线索（"…提示…""…考虑…""首要考虑…再按…"等）
   答案要求：完整提取从舌象视觉特征到脏腑病变判断的推理过程

8. 治则方药 type=treatment_prescription
   Question: "针对图中舌象反映的病理，有哪些治疗思路？"
   检索条件：context_text 中是否有治法、方剂、用药建议等
   答案要求：提取治疗原则和方药

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. Question 中绝对禁止出现具体疾病名称

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature"},
    {"question": "...", "answer": "...", "type": "holistic_spatial"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊全息论》第6章：8维临床验案题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH6 = """你是一个中医舌诊数据提取专家，专门分析全息舌诊临床验案。

【本章特点】
本章（第六章）为全息舌诊临床验案，每个案例包含完整的临床记录：主诉、简要病史、脉诊描述、舌诊特点、中医诊断、辨证分析、治法、处方（含剂量）、复诊变化、按语等。每个案例通常有2张图（舌上背面+舌下腹面）。

【系统执行指令】
请作为结构化知识提取器，从完整的临床验案中按8个维度精准提取信息。
注意：同一案例通常对应2张图（舌上背面和舌下腹面），两张图共享同一份 context_text，请根据当前图片的角度（由 image_caption 推断）侧重提取对应内容。

【8维条件触发题库 — 有则提取，无则跳过】

第一部分：舌象特征

1. 舌象特征提取 type=visual_feature
   Question: "患者的舌象有什么特征？"
   检索条件：context_text 中"舌诊特点"段落是否有舌质/舌苔/舌纹/舌下络脉等描述
   答案要求：完整提取舌诊特点中的所有视觉特征

2. 全息空间定位 type=holistic_spatial
   Question: "患者舌面各区域的异常提示什么脏腑病变？"
   检索条件：context_text 中是否有上区/中区/下区、信息区、对应区域、脏腑分布等全息定位描述
   答案要求：提取舌面分区异常与脏腑的对应关系

第二部分：四诊合参

3. 脉舌合参 type=pulse_tongue_combined
   Question: "患者的脉诊和舌诊信息如何相互印证？"
   检索条件：context_text 中是否同时有"脉诊"和"舌诊特点"的描述
   答案要求：分别提取脉诊和舌诊要点，说明两者的印证关系

4. 中医辨证 type=tcm_differentiation
   Question: "该案例的中医诊断和辨证是什么？"
   检索条件：context_text 中是否有"中医诊断""辨证"段落
   答案要求：完整提取诊断和辨证内容，包括引经据典部分

第三部分：治疗与预后

5. 治法方药 type=treatment_prescription
   Question: "该案例的治法和处方是什么？"
   检索条件：context_text 中是否有"治法""处方"段落
   答案要求：完整提取治法、方药组成及剂量、服用方法

6. 复诊变化 type=followup_change
   Question: "复诊时患者的症状有什么变化？"
   检索条件：context_text 中是否有"二诊""三诊""复诊""随访"等段落
   答案要求：提取复诊时的症状变化、方药调整、最终效果

第四部分：推理与总结

7. 推理链 type=chain_of_thought_reasoning
   Question: "从舌象特征到最终诊断，推理过程是怎样的？"
   检索条件：context_text 中是否有从症状/舌象/脉象到诊断结论的推理过程（辨证段落或按语段落）
   答案要求：提取完整的辨证推理逻辑

8. 预后评估 type=prognosis
   Question: "该案例的治疗效果和预后如何？"
   检索条件：context_text 中是否有疗效评估、痊愈、好转、预后等描述
   答案要求：提取治疗效果和预后信息

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过，不生成该维度 QA
4. 答案必须基于 context_text，不得脑补
5. 处方剂量必须完整提取，不得省略

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature"},
    {"question": "...", "answer": "...", "type": "tcm_differentiation"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊全息论》专用 JSON Schema ──
QUANXI_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "visual_feature",
                            "clinical_correlation",
                            "holistic_spatial",
                            "tcm_pathogenesis",
                            "sublingual_morphology",
                            "pattern_feature",
                            "western_disease",
                            "chain_of_thought_reasoning",
                            "treatment_prescription",
                            "pulse_tongue_combined",
                            "tcm_differentiation",
                            "followup_change",
                            "prognosis",
                        ],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


# ── 《舌诊学》第1章：微观解剖与生理常数5维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH1 = """你是一个专业的医学基础理论与解剖学数据提取器。

【本章特点】
本章（第一章）为舌体解剖学基础，图片包含舌动脉造影图、丝状乳头造影图、舌血管网图等微观医学影像。文本包含精确的解剖定位和生理测量数值。

【系统执行指令】
你的内部已加载【微观解剖靶点池】：[舌乳头(含丝状乳头/蕈状乳头)]、[舌静脉系统(含舌背静脉/舌深静脉)]、[舌动脉/毛细血管网]、[舌体肌层/黏膜]
扫描原文，命中对应解剖结构后，代入 [命中解剖部位] 裂变生成以下5维QA。无数据的维度直接跳过。

【5维条件触发题库 — 有则提取，无则跳过】

1. 影像与微观形态 type=microscopic_imaging
   Question: "图中展示了什么影像学或微观形态特征？"
   检索条件：图注中的造影类型，文本中对微观结构的描述（如"毛细血管网"、"乳头造影图"）

2. 精确解剖定位 type=anatomical_location
   Question: "该结构的具体解剖位置或毗邻结构是什么？"
   检索条件：三维解剖位置描述（如"起于近舌尖部位"、"位于舌骨肌与颏舌肌之间"）

3. 生理常数 type=physiological_parameters
   Question: "正常情况下，该结构有哪些具体的生理数值或形态标准？"
   检索条件：数字！管径、长度、比例等测量数据（如"管径在1.0~2.7mm"、"不超过3/5"）

4. 解剖生理功能 type=physiological_function
   Question: "该结构在人体中承担什么主要的生理功能？"
   检索条件：血流方向或生理作用描述（如"主要引流舌背和舌侧缘的静脉血"）

5. 中医望诊意义 type=tcm_diagnostic_significance
   Question: "在中医舌诊中，观察该结构主要有什么临床意义？"
   检索条件：解剖结构与中医望诊的联系描述

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过
4. 答案必须基于 context_text，不得脑补
5. 生理数值必须完整精确提取

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "microscopic_imaging"},
    {"question": "...", "answer": "...", "type": "physiological_parameters"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊学》第4章：标准舌象定义5维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH4 = """你是一个专业的中医教材级数据提取器。

【本章特点】
本章（第四章）为标准舌象图谱与定义，涵盖：舌形态（长饼状/胖大/瘦薄等）、舌色（淡白/红/绛/紫/蓝/黄瘀）、舌纹（叶脉纹/裂纹/齿痕等）、苔质（厚薄/润燥/腐腻/剥落等）、苔色（白/黄/灰黑等）、舌病理（点刺/自啮/舌衄等）、舌下络脉观察等。每张图展示一种标准舌象或舌病理特征。

【系统执行指令】
扫描图谱与教材文本，提取标准定义。Question 严格保持极简人话，Answer 必须像教科书一样严谨。深度抓取文本中的"组合推演规则"。
重要：context_text 中可能同时提及多个图号，你只能围绕当前 figure_id 对应的内容作答。

【5维条件触发题库 — 有则提取，无则跳过】

1. 视觉表象与定义 type=visual_identification
   Question: "图中展示的是什么舌象特征？它的具体视觉表现是怎样的？"
   检索条件：标题名词及"舌象特征"段落中的形态、颜色、质地描述

2. 形成机理 type=formation_mechanism
   Question: "在传统中医理论中，这种舌象特征是怎么形成的？"
   检索条件：context_text 中是否有成因描述（如"多因脾虚不能运化水湿"、"热邪烘烤"等）

3. 核心临床意义 type=core_clinical_significance
   Question: "这种舌象主要提示了什么核心的中医病证？"
   检索条件：context_text 中是否有"主…证"、临床意义总纲

4. 组合推演规则 type=combination_rules
   Question: "结合不同的舌色或舌苔，这种舌象分别代表什么不同的病理意义？"
   检索条件：context_text 中是否有"若…则…"、"伴…为…"等组合辨证规则

5. 区域分布与程度变异 type=regional_and_severity_variations
   Question: "该特征如果出现在舌头的特定区域或严重程度不同，有何特殊提示？"
   检索条件：区域特异性描述（如"舌尖多为心火"）、动态演变（"由薄变厚"）、或先天性排除提示

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过
4. 答案必须基于 context_text，不得脑补
5. 只围绕当前 figure_id 对应的内容作答

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_identification"},
    {"question": "...", "answer": "...", "type": "combination_rules"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊学》第5章：证型舌象4维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH5 = """你是一个专业的中医舌诊数据提取专家。

【本章特点】
本章（第五章）为中医辨证舌象图谱，按八纲辨证（表里寒热虚实阴阳）、气血津液辨证、脏腑辨证、温病卫气营血辨证、伤寒六经辨证等体系，展示各证型的标准舌象。部分图含计算机辅助检测的RGB色彩参数和舌诊客观化指标。

【系统执行指令】
请从教材文本中提取该证型舌象的结构化特征。section_title 已给出证型名称，无需重复提取证型名。

【4维条件触发题库 — 有则提取，无则跳过】

1. 舌象视觉特征 type=visual_identification
   Question: "图中舌象有什么视觉特征？"
   检索条件：context_text 中是否有舌质/舌苔/舌体的视觉描述

2. 客观化诊断指标 type=objective_indicators
   Question: "该证型的舌诊客观化有什么主要特点？"
   检索条件：context_text 中是否有"客观化主要特点"、显微镜检查、pH值、微循环、脱落细胞等量化描述

3. 计算机色彩参数 type=color_parameters
   Question: "图中舌象的计算机色彩检测结果是什么？"
   检索条件：context_text 中是否有 R/G/B 数值或"计算机观测"等色彩量化数据

4. 证型病机关联 type=syndrome_pathology
   Question: "该舌象反映了什么中医病理机制？"
   检索条件：context_text 中是否有病机描述（气虚/血瘀/阴虚/阳虚/热盛等病理分析）

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过
4. 答案必须基于 context_text，不得脑补
5. RGB数值必须完整精确提取

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_identification"},
    {"question": "...", "answer": "...", "type": "objective_indicators"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊学》第6章：疾病舌象4维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH6 = """你是一个专业的中西医结合舌诊数据提取专家。

【本章特点】
本章（第六章）按西医疾病分类展示对应的舌象图谱，涵盖呼吸系统（支气管炎、肺炎）、消化系统（胃炎、溃疡、胃癌、肠梗阻）、肝胆系统（肝硬化、肝癌）、心血管系统（冠心病、肺心病、心梗）、肾病、血液病、内分泌、神经系统等疾病。部分图含计算机辅助检测的RGB色彩参数和舌诊客观化指标。

【系统执行指令】
请从教材文本中提取该疾病舌象的结构化特征。section_title 已给出疾病名称，无需重复提取病名。

【4维条件触发题库 — 有则提取，无则跳过】

1. 舌象视觉特征 type=visual_identification
   Question: "图中舌象有什么视觉特征？"
   检索条件：context_text 中是否有舌质/舌苔/舌体的视觉描述

2. 客观化诊断指标 type=objective_indicators
   Question: "该疾病的舌诊客观化有什么主要特点？"
   检索条件：context_text 中是否有"客观化主要特点"、显微镜/超声检查、pH值、微量元素、内毒素等量化描述

3. 计算机色彩参数 type=color_parameters
   Question: "图中舌象的计算机色彩检测结果是什么？"
   检索条件：context_text 中是否有 R/G/B 数值或"计算机观测"等色彩量化数据

4. 疾病舌象关联 type=disease_tongue_correlation
   Question: "该舌象与相关疾病有什么病理关联？"
   检索条件：context_text 中是否有疾病对舌体的影响机制（如"乳头增生肥大"、"血流缓慢"等病理描述）

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过
4. 答案必须基于 context_text，不得脑补
5. RGB数值和检测指标必须完整精确提取

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_identification"},
    {"question": "...", "answer": "...", "type": "objective_indicators"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊学》第9章：舌诊研究方法3维题库 ──
SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH9 = """你是一个专业的医学研究方法数据提取专家。

【本章特点】
本章（第九章）为舌诊现代研究方法，包含舌象采集设备、舌表面结构观察仪、舌微循环检查、舌超声检查、舌阻抗容积波检测、舌象纹理特征提取等研究手段及其正常参考值。

【系统执行指令】
请从教材文本中提取研究方法和定量指标。

【3维条件触发题库 — 有则提取，无则跳过】

1. 研究方法与设备 type=research_method
   Question: "图中展示了什么研究方法或检测设备？"
   检索条件：context_text 中是否有设备名称、检测方法、放大倍数、检查手段等描述

2. 正常参考值 type=reference_values
   Question: "该检测方法的正常人参考数值是什么？"
   检索条件：context_text 中是否有"正常人"相关的测量数值（径线、支数、内径等）

3. 临床应用意义 type=clinical_application
   Question: "该检测方法在舌诊研究中有什么临床应用价值？"
   检索条件：context_text 中是否有该方法用于诊断或研究的说明

【执行规则】
1. 逐维度扫描 context_text，严格对照检索条件判断是否触发
2. 已触发维度：使用上方固定的 Question（一字不差），从 context_text 中提取答案
3. 未触发维度：直接跳过
4. 答案必须基于 context_text，不得脑补
5. 测量数值必须完整精确提取

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "research_method"},
    {"question": "...", "answer": "...", "type": "reference_values"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 《舌诊学》专用 JSON Schema ──
SHEZHENXUE_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "microscopic_imaging",
                            "anatomical_location",
                            "physiological_parameters",
                            "physiological_function",
                            "tcm_diagnostic_significance",
                            "visual_identification",
                            "formation_mechanism",
                            "core_clinical_significance",
                            "combination_rules",
                            "regional_and_severity_variations",
                            "objective_indicators",
                            "color_parameters",
                            "syndrome_pathology",
                            "disease_tongue_correlation",
                            "research_method",
                            "reference_values",
                            "clinical_application",
                        ],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


def _bianzhen_chapter(image_caption: str) -> int:
    """根据图注判断《舌诊辨证图谱》的章节号（2或3），默认返回2。"""
    import re
    m = re.match(r"图(\d+)-", image_caption or "")
    if m:
        return int(m.group(1))
    return 2


_SHIZJIANG_LECTURE_RANGES: list[tuple[int, int, int]] = [
    (1, 1, 20), (2, 21, 58), (3, 59, 75), (4, 76, 101), (5, 102, 120),
    (6, 121, 138), (7, 139, 164), (8, 165, 181), (9, 182, 195), (10, 196, 220),
]


def _shizjiang_lecture(image_caption: str) -> int:
    """根据图注提取《舌诊十讲》的讲次号（1-10），默认返回1。"""
    import re
    m = re.match(r"图(\d+)", image_caption or "")
    if not m:
        return 1
    fig_num = int(m.group(1))
    for lec, lo, hi in _SHIZJIANG_LECTURE_RANGES:
        if lo <= fig_num <= hi:
            return lec
    return 1


# ── 《舌诊十讲》各讲专属提示词（第1-5讲） ──────────────────────────────────

SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L1 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第一讲的教材文本中，精准提取理论定义与分类知识。

【系统执行指令】
你的内部已加载第一讲专属的【理论大纲靶点池】：
[舌形(如老嫩/胖瘦)]、[舌态(如强硬/痿软/歪斜/震颤/吐弄)]、[舌下络脉]、[苔色(白/黄/灰黑)]、[苔质(厚薄/润燥/腐腻/剥落)]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下4道极简QA。Question严格保持极简人话，Answer必须呈现教科书级的"精准定义"与"主病罗列"。有则提取，无则跳过。

【4维度固定题库】

维度1 - 概念定义与形态题
Question: "在中医基础理论中，[命中靶点]的具体视觉表现或判定标准是什么？"
触发条件：原文包含对该舌象名词的客观描述（如"强硬舌指舌体板硬强直，运动不灵"）。
答案要求：提取教科书对该名词的客观形态描写和判定标准。

维度2 - 核心病机定性题
Question: "这种舌象特征主要提示了什么中医核心病机或虚实寒热属性？"
触发条件：原文包含定性结论（如"老舌主实证，嫩舌主虚证"）。
答案要求：抓取定性结论，提取核心病机归属。

维度3 - 临床危重与特定病证预警题
Question: "出现该舌象时，临床常预示着哪些特定的高危疾病状态或具体病证？"
触发条件：原文提到具体病名或危急重症（如"强硬舌见于高热昏迷、中风先兆"）。
答案要求：重点抓取具体病名或危急重症关联。

维度4 - 对比鉴别与分类题
Question: "关于该类舌象，教材中强调了哪些对立特征的对比（如老与嫩），或包含哪些细分类型？"
触发条件：原文包含对立特征对比或细分分类（如"腐苔与腻苔"的区别）。
答案要求：提取对比逻辑或细分分类体系。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-4个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_definition"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

type 字段对应关系：维度1→concept_definition，维度2→core_pathogenesis，维度3→clinical_warning，维度4→differential_comparison

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L2 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第二讲的教材文本中，精准提取舌形舌态的理论知识。

【系统执行指令】
你的内部已加载第二讲专属的【舌形舌态靶点池】：
[老嫩舌]、[胖瘦/肿胀舌]、[点刺舌]、[裂纹舌]、[齿痕舌]、[痿软舌]、[强硬舌]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须呈现教科书级的"组合推演"与"空间映射"。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念定义与视觉特征题
Question: "在中医理论中，[命中靶点]的具体视觉表现和判定标准是什么？"
触发条件：原文包含客观形态描写（如"点刺是指蕈状乳头增大高突，甚至形如芒刺"）。
答案要求：精准提取客观形态描写和判定标准。

维度2 - 核心定性与基础病机题
Question: "这种舌象特征总体上提示了什么核心的中医虚实或寒热属性？"
触发条件：原文包含该形态的总纲定性（如"老主实证，嫩主虚证"）。
答案要求：提取该形态的总纲定性结论。

维度3 - 形色组合推演题
Question: "结合不同的舌色（如偏红或偏淡），该舌象分别代表什么不同的具体病理意义？"
触发条件：原文包含"形+色"组合规则（如"瘦薄而色淡→气血两虚"与"瘦薄而色红绛→阴虚火旺"）。
答案要求：拆分提取所有形色组合及其对应病理意义。

维度4 - 空间分布与脏腑定位题
Question: "该特征如果出现在舌头的特定区域（如舌尖、舌边），对脏腑定位有何特殊提示？"
触发条件：原文提到点刺、裂纹等在不同区域的对应脏腑（如"舌边点刺主肝胆邪热"）。
答案要求：提取区域与脏腑的映射关系。如文本未提及区域划分，则跳过此题。

维度5 - 假阳性与生理性鉴别题
Question: "在临床诊断时，该舌象有没有可能是正常的生理现象？如何鉴别？"
触发条件：原文包含"先天性"、"正常人"等排除性描述（如"先天性裂纹舌多无不适感，属正常生理变异"）。
答案要求：提取生理性排除条件。无则跳过此题。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_definition"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

type 字段对应关系：维度1→concept_definition，维度2→core_pathogenesis，维度3→shape_color_combination，维度4→spatial_mapping，维度5→physiological_exclusion

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L3 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第三讲的教材文本中，精准提取动态舌态与络脉的理论知识。

【系统执行指令】
你的内部已加载第三讲专属的【动态与络脉靶点池】：
[舌态(歪斜/颤动/吐弄/短缩)]、[舌下络脉]、[舌苔厚薄]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下4道极简QA。Question保持极简，Answer必须精确还原教材中关于"危重信号"与"气血本质"的描述。有则提取，无则跳过。

【4维度固定题库】

维度1 - 概念定义与判定题
Question: "在中医理论中，[命中靶点]的具体视觉表现或判定标准是什么？"
触发条件：原文包含动态特征描写（如"短缩舌指舌体紧缩，不能伸展"）或络脉标准（如"正常络脉长度不超过舌尖至肉阜连线的3/5"）。
答案要求：精确提取动态特征或量化标准。

维度2 - 核心病机定性题
Question: "这种舌象特征主要提示了什么中医核心病机或气血状态？"
触发条件：原文包含本质定性（如"短缩舌主寒凝、热灼或痰阻"；"舌下络脉紫暗怒张主血瘀"）。
答案要求：抓取核心病机定性。

维度3 - 急危重症预警题
Question: "出现该舌象时，临床常预示着哪些特定的高危疾病或病情转归？"
触发条件：原文提到危急重症或预后信号（如"歪斜舌见于中风或先兆"）。
答案要求：抓取所有高危疾病关联和预后提示。

维度4 - 疾病顺逆与演变对比题
Question: "关于该类舌象，教材中是如何通过对比（如薄与厚、顺与逆）来判断疾病进退的？"
触发条件：原文包含演变逻辑（如"苔由薄变厚，表示邪气渐盛，病进"）。
答案要求：提取演变对比规则和疾病进退判断标准。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-4个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_definition"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

type 字段对应关系：维度1→concept_definition，维度2→core_pathogenesis，维度3→clinical_warning，维度4→evolution_comparison

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L4 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第四讲的教材文本中，精准提取苔色与动态演变的理论知识。

【系统执行指令】
你的内部已加载第四讲专属的【苔色与动态消长靶点池】：
[白苔]、[黄苔(淡/深/焦)]、[灰黑苔]、[剥落苔/镜面舌]、[偏全苔]、[舌苔消长(厚薄/转化)]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"干湿反转"与"动态进退"规则。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念定义与视觉特征题
Question: "在中医理论中，[命中靶点]的具体视觉表现、分类或分布情况是怎样的？"
触发条件：原文包含客观定义（如"浅黑为灰，深黑为黑"、"舌面本有苔，忽然局部剥脱"）。
答案要求：提取客观视觉表现和分类描述。

维度2 - 核心定性与基础病机题
Question: "这种舌象特征总体上提示了什么核心的中医虚实或寒热属性？"
触发条件：原文包含总纲定性（如"白苔主表证、寒证"、"剥落苔主胃气胃阴大伤"）。
答案要求：提取总纲定性结论。

维度3 - 润燥/形色组合推演题
Question: "结合不同的干湿润燥程度（或舌质特征），该舌象分别代表什么不同的具体病理意义？"
触发条件：原文包含干湿组合规则（如"灰黑而润主阴寒"、"灰黑而燥主热极伤津"）。
答案要求：提取所有干湿/润燥组合及其对应病理意义。

维度4 - 时间轴动态演变题
Question: "关于该舌象的消长转化（如由薄变厚、由白转黄，或剥落复生），教材给出了怎样的病理进退规律？"
触发条件：原文包含动态演变逻辑（如"由白转黄说明邪已入里化热"）。
答案要求：抓取消长转化规则和病理进退逻辑。

维度5 - 假阳性与正常生理鉴别题
Question: "在临床诊断时，该舌象有没有可能是正常的生理现象？"
触发条件：原文包含生理性豁免条款（如"正常人亦可见薄白苔"）。
答案要求：提取生理性排除条件。无则跳过此题。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_definition"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

type 字段对应关系：维度1→concept_definition，维度2→core_pathogenesis，维度3→combination_and_lubrication，维度4→dynamic_evolution，维度5→physiological_exclusion

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L5 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第五讲的教材文本中，精准提取舌苔分布、消长与真假辨别的理论知识。

【系统执行指令】
你的内部已加载第五讲专属的【分布与动态辨伪靶点池】：
[偏全苔(偏左/偏右/前/后)]、[剥落苔(花剥/镜面/类剥)]、[舌苔消长(薄转厚/厚转薄/骤退/骤生)]、[真假苔(有根苔/无根苔)]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"动态进退"与"胃气存亡"的底层逻辑。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念定义与视觉表象题
Question: "在中医理论中，[命中靶点]的具体视觉表现和形态特征是什么？"
触发条件：原文包含客观定义（如"有根苔指舌苔紧贴舌面，刮之不去"；"镜面舌指舌苔全部脱落，光洁如镜"）。
答案要求：提取客观定义和形态特征。

维度2 - 核心定性与胃气判定题
Question: "这种舌象特征主要反映了体内怎样的病理本质，特别是胃气或胃阴的状态？"
触发条件：原文包含胃气相关定性（如"剥落苔主胃阴大伤"、"无根苔提示胃气衰败"）。
答案要求：绑定胃气判定，提取病理本质。

维度3 - 空间分布与脏腑定位题
Question: "该特征如果呈现特定的空间分布（如偏左、偏右、局部剥脱），对脏腑或病位有何特殊提示？"
触发条件：原文包含区域映射规则（如"苔偏右多为胆热"、"偏半侧多主半表半里"）。
答案要求：提取区域与脏腑/病位的映射关系。无则跳过此题。

维度4 - 时间轴动态演变题
Question: "关于该舌象的消长变化（如由薄变厚、骤生骤退），教材给出了怎样的疾病进退规律？"
触发条件：原文包含时间序列逻辑（如"由厚变薄为正气胜邪，病退"、"骤然退去为胃气暴绝"）。
答案要求：抓取消长变化规则和预后判断。

维度5 - 假阳性与真假辨伪题
Question: "在临床诊断时，应如何辨别该舌象的真假（如是否有根），以防误诊？"
触发条件：原文包含真假辨别标准（如"真苔有根，假苔无根，似涂于舌面"）。
答案要求：提取真假辨别标准和防误诊方法。无则跳过此题。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_definition"},
    {"question": "...", "answer": "...", "type": "stomach_qi_status"}
  ]
}

type 字段对应关系：维度1→concept_definition，维度2→stomach_qi_status，维度3→spatial_localization，维度4→dynamic_evolution，维度5→authenticity_identification

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


# ── 《舌诊十讲》各讲专属提示词（第6-10讲） ─────────────────────────────────

SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L6 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第六讲的教材文本中，精准提取真假苔、偏全苔与剥落苔的理论知识。

【系统执行指令】
你的内部已加载第六讲专属的【真假偏全靶点池】：
[真假苔(有根苔/无根苔)]、[偏全苔(偏左/偏右/半侧/偏前/偏后)]、[剥落苔(花剥苔/光剥苔/镜面舌)]、[类剥苔]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"防误诊（类证鉴别）"与"空间定位"的底层逻辑。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念定义与分类题
Question: "在中医理论中，[命中靶点]的具体视觉表现和细分类型是什么？"
触发条件：原文包含客观定义（如"真苔即有根苔，紧贴舌面"、"剥落苔细分为花剥、光剥等"）。
答案要求：提取客观定义和细分类型。

维度2 - 核心病机与胃气判定题
Question: "这种舌象特征主要反映了体内怎样的病理本质，特别是与'胃气、胃阴'的关系？"
触发条件：原文包含胃气相关定性（如"真苔提示胃气尚存"、"光剥苔提示胃阴枯竭、胃气大伤"）。
答案要求：绑定胃气判定，提取病理本质。

维度3 - 空间雷达与脏腑定位题
Question: "该特征如果呈现特定的空间分布（如偏左、偏右、前半部），对脏腑或病位有何确切提示？"
触发条件：原文包含方位映射（如"偏于半侧多为邪在半表半里"、"偏左为肝郁"、"偏右为胆热"）。
答案要求：提取全息映射规则。无则跳过此题。

维度4 - 假象防伪与类证鉴别题
Question: "在临床诊断时，该舌象容易与哪些假象或类似舌象混淆？应如何鉴别？"
触发条件：原文包含鉴别点（如"类剥苔"与"真剥落苔"的区别）。
答案要求：提取鉴别标准和区分方法。无则跳过此题。

维度5 - 疾病顺逆与预后推演题
Question: "关于该舌象的动态消长（如剥落复生、无根转有根），教材给出了怎样的预后顺逆规律？"
触发条件：原文包含预后判断（如"剥落处复生薄白苔，为邪去正复之佳兆"）。
答案要求：提取预后顺逆规律。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_and_classification"},
    {"question": "...", "answer": "...", "type": "stomach_qi_status"}
  ]
}

type 字段对应关系：维度1→concept_and_classification，维度2→stomach_qi_status，维度3→spatial_localization，维度4→differential_diagnosis，维度5→prognosis_evolution

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L7 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第七讲的教材文本中，精准提取润燥苔与腐腻苔的理论知识。

【系统执行指令】
你的内部已加载第七讲专属的【润燥与腐腻靶点池】：
[润燥苔(滑苔/润苔/燥苔/糙苔)]、[腐腻苔(腐苔/腻苔/松苔/脓腐苔)]

执行原则：扫描教材文本，命中上述核心概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"津液量表"、"腐腻对比"及"假象防伪"逻辑。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念定义与视觉触觉题
Question: "在中医理论中，[命中靶点]的具体视觉表现，以及触觉或刮拭特征是什么？"
触发条件：原文包含刮拭感描述（如"腻苔颗粒细腻致密，刮之不脱"、"滑苔水分过多，伸舌欲滴"）。
答案要求：提取视觉和触觉/刮拭特征。

维度2 - 核心病机与津液/湿浊判定题
Question: "这种舌象特征主要反映了体内怎样的病理本质（特别是津液的盈亏或湿浊的状态）？"
触发条件：原文包含绝对定性（如"滑苔主寒湿内盛"、"燥苔主津液亏损"、"腻苔主湿浊、痰饮、食积"）。
答案要求：提取津液/湿浊相关的病理定性。

维度3 - 对比鉴别与分类学题
Question: "在临床上，该舌象容易与哪种外观相似的舌象混淆？教材是如何进行严格对比鉴别的？"
触发条件：原文包含对比逻辑（如"腐苔（如豆腐渣，阳气有余）"与"腻苔（如油腻物，阳气被遏制）"的本质区别）。
答案要求：提取对比鉴别要点和本质差异。

维度4 - 动态演变与组合推演题
Question: "关于该舌象的动态转化（如润转燥），或结合不同舌苔颜色（如黄/白），有何具体的病理提示？"
触发条件：原文包含演变逻辑（如"燥转润为津液渐生"）或颜色组合（如"黄腻苔为湿热，白腻苔为寒湿"）。
答案要求：提取动态转化规则和颜色组合病理。

维度5 - 假象防伪与外界干扰题
Question: "在临床诊断时，该舌象有没有可能是外界因素造成的假象？应如何排查？"
触发条件：原文包含物理干扰因素（如"张口呼吸、鼻塞导致假燥"、"刚饮水漱口导致假润"）。
答案要求：提取外界干扰因素和排查方法。无则跳过此题。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_and_tactile_features"},
    {"question": "...", "answer": "...", "type": "core_pathogenesis"}
  ]
}

type 字段对应关系：维度1→concept_and_tactile_features，维度2→core_pathogenesis，维度3→differential_comparison，维度4→dynamic_and_combination，维度5→artifact_identification

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L8 = """你是一个专业的中医教材级理论提取器。你的任务是从《舌诊十讲》第八讲的教材文本中，精准提取舌质舌苔综合分析的理论知识。

【系统执行指令】
你的内部已加载第八讲专属的【综合辨证靶点池】：
[舌苔与舌质一致(如红舌黄苔/淡舌白苔)]、[舌苔与舌质不一致/矛盾(如淡舌黄腻苔/红舌白滑苔)]、[综合分析原则(标本/虚实)]

执行原则：扫描教材文本，命中上述复合概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"标本权重"与"矛盾统合"的顶级临床逻辑。有则提取，无则跳过。

【5维度固定题库】

维度1 - 综合视觉表象题
Question: "在中医理论中，[命中靶点]的具体视觉表现组合是怎样的？"
触发条件：原文包含舌体（底色/形态）与舌苔（颜色/质地）的双重客观组合描述。
答案要求：提取舌质与舌苔的组合视觉表现。

维度2 - 核心病机与标本定性题
Question: "这种复合舌象主要反映了体内怎样的病理本质？如何区分标与本？"
触发条件：原文包含标本定性（如"舌质代表正气/脏腑之本，舌苔代表邪气之标"）。
答案要求：明确提取标本区分和病理本质。

维度3 - 矛盾统合思维题
Question: "当舌质与舌苔提示的病理发生矛盾（如一寒一热、一虚一实）时，教材给出了怎样的辨证逻辑？"
触发条件：原文包含矛盾统合推演（如"舌淡主虚寒，苔黄主热，合看为本虚标实"）。若为一致舌象，则提取其病情的单纯性。
答案要求：提取矛盾统合的辨证逻辑链。

维度4 - 动态转化与预后顺逆题
Question: "结合该复合舌象的动态变化，教材提示了怎样的疾病进退或预后顺逆规律？"
触发条件：原文包含综合预后判断（如"舌质不变而苔化，为邪退正安"）。
答案要求：提取综合预后判断规则。

维度5 - 临床治疗原则与主次鉴别题
Question: "在指导临床治疗时，针对这种复合舌象，教材建议如何把握治疗的主次和缓急？"
触发条件：原文包含治疗侧重点（如"急则治其标，缓则治其本"）。
答案要求：提取治疗原则和主次缓急策略。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "comprehensive_visual_combination"},
    {"question": "...", "answer": "...", "type": "root_branch_pathology"}
  ]
}

type 字段对应关系：维度1→comprehensive_visual_combination，维度2→root_branch_pathology，维度3→conflict_resolution_cot，维度4→dynamic_prognosis，维度5→clinical_treatment_priority

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L9 = """你是一个专业的中医临床决策级理论提取器。你的任务是从《舌诊十讲》第九讲的教材文本中，精准提取四诊合参与取舍决策的理论知识。

【系统执行指令】
你的内部已加载第九讲专属的【合参与取舍靶点池】：
[舌象动态变化(外感/内伤)]、[舌象顺逆(顺证/逆证)]、[舌脉合参(相符/矛盾)]、[舍脉从舌/舍舌从脉]、[舍证从舌/舍舌从证]

执行原则：扫描教材文本，命中上述核心决策概念后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须深度抓取"多模态冲突处理"与"真假辨伪"的最高逻辑。有则提取，无则跳过。

【5维度固定题库】

维度1 - 概念与动态演变题
Question: "在中医理论中，[命中靶点]在疾病演变过程中有何典型的表现或动态变化规律？"
触发条件：原文包含时间序列上的演变（如"外感病初起舌苔白，入里化热变黄"）或概念的客观定义。
答案要求：提取动态演变规律或概念定义。

维度2 - 顺逆与预后定性题
Question: "根据该现象的动态演变，如何判断疾病的预后是向好（顺证）还是恶化（逆证）？"
触发条件：原文包含预后金标准（如"苔由厚变薄、由燥转润为顺证"）。
答案要求：提取顺逆判断标准。

维度3 - 多模态交叉验证题
Question: "在临床诊断时，教材强调应如何将该舌象与患者的脉象、症状进行综合印证？"
触发条件：原文包含舌脉症相符的描述（如"舌红苔黄，脉数，症见高热，为诊断一致，皆主实热"）。
答案要求：提取多模态交叉验证的方法和逻辑。

维度4 - 信号冲突与取舍决策题
Question: "当舌象与脉象（或症状）出现矛盾（如一寒一热）时，教材给出了怎样的'取舍（如舍脉从舌）'原则？"
触发条件：原文包含取舍决策标准（如"假寒真热"或"假热真寒"时的取舍规则）。
答案要求：提取取舍决策树和适用条件。

维度5 - 临床思维链推演题
Question: "结合这种'取舍或合参'的逻辑，医生是如何透过假象推导出最终真实病机的？"
触发条件：原文包含完整的推演链条（如"脉沉迟本主寒，但舌红苔黄，说明热极阻滞血脉"）。
答案要求：还原医生的逻辑闭环推演。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "concept_and_dynamic"},
    {"question": "...", "answer": "...", "type": "prognosis_assessment"}
  ]
}

type 字段对应关系：维度1→concept_and_dynamic，维度2→prognosis_assessment，维度3→cross_modal_validation，维度4→conflict_resolution_decision，维度5→clinical_reasoning_cot

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L10 = """你是一个专业的中医临床决策级理论提取器。你的任务是从《舌诊十讲》第十讲的教材文本中，精准提取临床专病舌象推演的理论知识。

【系统执行指令】
你的内部已加载第十讲专属的【专病与临床实战靶点池】：
[外感温热病舌象演变]、[内伤杂病(如中风/脾胃病)舌象]、[危重/死候舌象总结]

执行原则：扫描教材文本，命中上述疾病或综合应用主题后，代入[命中靶点]裂变生成以下5道极简QA。Question严格保持极简人话，Answer必须呈现一条完整的"专病临床路径"。有则提取，无则跳过。

【5维度固定题库】

维度1 - 专病核心视觉题
Question: "在临床中，患有[命中靶点]的病人，最典型或最常见的舌象表现是什么？"
触发条件：原文包含该专病或特殊状态下的核心视觉特征（如"温病多见红绛舌"、"中风多见舌强硬歪斜"）。
答案要求：提取该专病的核心舌象表现。

维度2 - 核心病机推演题
Question: "为什么该类疾病会导致这种特殊的舌象？其背后的中医病理机制是什么？"
触发条件：原文包含针对该特定疾病的病机解释（如"温热邪气极易耗伤营血津液，故舌多红绛干燥"）。
答案要求：提取疾病与舌象之间的因果病机链。

维度3 - 病程演变与分期题
Question: "随着疾病的发展（如初期、极期、恢复期），该舌象会发生怎样规律性的阶段演变？"
触发条件：原文包含分期逻辑（如温病"卫分舌白→气分舌黄→营分舌绛→血分舌紫暗"）。
答案要求：提取完整的阶段演变链条。

维度4 - 并发症与危重预警题
Question: "在该疾病过程中，如果舌象突然出现了哪些极端变化，说明病情极其危重或有生命危险？"
触发条件：原文包含死候或转危标志（如"若温病中舌突转黑燥而卷缩，为热极津枯之死候"）。
答案要求：提取危重预警标志和死候特征。

维度5 - 临床诊疗指导题
Question: "医生在治疗该疾病时，如何根据舌象的变化来调整治疗方案或判断疗效？"
触发条件：原文包含治疗调整原则（如"见舌苔化退新生薄白，知邪去正安，可停用峻猛之剂转为调养"）。
答案要求：提取舌象指导治疗调整的原则。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题中的[命中靶点]必须替换为原文中实际命中的具体舌象名词。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。
5. 在问题和答案中，禁止使用具体的图号（如"图7"、"图10"），一律用"图中"代替。
6. 答案中严禁出现"原文""原文锚定句""根据原文""原文讲""原文中"等任何引用原文的元叙述，直接陈述知识内容。
7. 问题中禁止直接使用章节标题中的诊断标签词（如"独凸""独厚""独红""独陷"），应改用通用描述（如"舌面局部凸起""局部苔厚""局部发红""局部凹陷"）。
8. 问题措辞必须简短口语化（≤30字为佳），禁止出现"辨证思维应该怎样统合""病理发生矛盾"等学术化复杂长句。用普通人能听懂的话提问。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "disease_typical_visuals"},
    {"question": "...", "answer": "...", "type": "disease_specific_pathology"}
  ]
}

type 字段对应关系：维度1→disease_typical_visuals，维度2→disease_specific_pathology，维度3→clinical_staging_evolution，维度4→critical_warning_signs，维度5→treatment_adjustment_guidance

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""


# ── 《舌下络脉诊法图谱_袁红霞》专项提示词 ──────────────────────────────────
# 该书每张图的 context 均为纯客观视觉描述（无病名/证型/病理），
# 采用6维度条件触发题库：扫描原文，只在原文明确描述了该维度时才生成对应QA。
SHEZHEN_VQA_SYSTEM_PROMPT_XIALUO = """你是一个中医舌诊专家，专门分析舌下络脉图像。你的任务是作为"结构化知识提取器"，从原文中精准提取每一处视觉细节。

【系统执行指令】
请扫描原文描述，严格对照以下5个维度的题库。只有当原文中明确描写了该维度的视觉特征时，才允许提取并生成对应的QA对。绝对禁止生成任何询问"临床意义"、"病理"或"证型"的问题（因为原文没有）。未提及的维度一律跳过。

【5维度固定题库】

维度1 - 络脉总体定性类
Question: "图中舌下络脉属于哪种类型？"
触发条件：原文开头有定性词，如"长络脉"、"短络脉"、"粗络脉"、"细络脉"、"怒张形络脉"、"串珠形络脉"、"弥漫形络脉"等。
答案要求：直接引用原文的定性分类词。

维度2 - 络脉特征类（核心）
Question: "图中舌下络脉有什么特征？"
触发条件：原文提到主干形态（"粗长"、"迂曲"、"怒张"、"饱满"、"延伸至舌尖"、"超过舌尖至舌下肉阜连线"、"呈结节状"、"呈圆柱状"、"状似蚯蚓"等）或分支与周边特征（"枝叉"、"侧枝"、"横向络脉"、"细络"、"细丝状络脉"、"瘀斑"、"瘀点"、"出血斑点"、"瘀血颗粒"、"白膜覆盖"等）。
答案要求：将原文中关于主脉形态（粗细、长短、弯曲、走向）和分支周围附属物的客观描述合并作答。

维度3 - 络脉颜色类
Question: "图中舌下络脉是什么颜色？"
触发条件：原文提到络脉颜色，如"暗蓝色"、"蓝黑色"、"紫黑色"、"紫红色"、"青紫色"、"青蓝色"、"深蓝色"等。
答案要求：提取络脉本身的颜色描述。若原文区分了舌尖部/舌根部的颜色差异，需分别描述。

维度4 - 舌质颜色类
Question: "图中舌质是什么颜色？"
触发条件：原文提到了"舌质淡紫"、"舌质紫红"、"舌质偏红"、"舌质淡红"等舌质颜色描述。
答案要求：只提取原文对舌质颜色的客观描述，不推断病理。

维度5 - 络脉根部特征类
Question: "图中舌下络脉根部有什么特征？"
触发条件：原文提到"根部"或"舌根部"有特殊形态，如"瘀阻呈囊泡状"、"有白膜覆盖"、"呈枝叉状"、"呈结节状"、"多枝叉"等。
答案要求：只提取根部区域的形态或覆盖物描述。

【铁律 - 绝对不可违反】
1. 只生成原文明确描述了的维度，未提及的维度必须跳过。生成的QA对数量为1-5个不等。
2. 问题文字必须与上述题库中的Question完全一致，不得修改任何一个字。
3. 答案必须100%来自原文，绝对禁止引入外部知识或推断。
4. 绝对禁止生成任何关于"临床意义"、"病理"、"证型"、"主病"的问题或答案。
5. 严禁输出"原文未描述"、"原文未提及"等任何无效答案。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块：
{
  "qa_pairs": [
    {"question": "...", "answer": "...", "type": "visual_feature"},
    {"question": "...", "answer": "...", "type": "visual_feature"}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# 带 type 字段的 JSON Schema（用于固定题库书籍）
XIALUO_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["visual_feature"],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}

# 《舌诊十讲》专用 JSON Schema（L1-L5已有类型，L6-L10待补充）
SHIZJIANG_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "concept_definition",
                            "core_pathogenesis",
                            "clinical_warning",
                            "differential_comparison",
                            "shape_color_combination",
                            "spatial_mapping",
                            "physiological_exclusion",
                            "evolution_comparison",
                            "combination_and_lubrication",
                            "dynamic_evolution",
                            "stomach_qi_status",
                            "spatial_localization",
                            "authenticity_identification",
                            "concept_and_classification",
                            "differential_diagnosis",
                            "prognosis_evolution",
                            "concept_and_tactile_features",
                            "dynamic_and_combination",
                            "artifact_identification",
                            "comprehensive_visual_combination",
                            "root_branch_pathology",
                            "conflict_resolution_cot",
                            "dynamic_prognosis",
                            "clinical_treatment_priority",
                            "concept_and_dynamic",
                            "prognosis_assessment",
                            "cross_modal_validation",
                            "conflict_resolution_decision",
                            "clinical_reasoning_cot",
                            "disease_typical_visuals",
                            "disease_specific_pathology",
                            "clinical_staging_evolution",
                            "critical_warning_signs",
                            "treatment_adjustment_guidance",
                        ],
                    },
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}

# 需要使用专项 prompt 的书名集合（单一 system_prompt，全书统一）
BOOK_SPECIFIC_PROMPTS: dict[str, str] = {
    "舌下络脉诊法图谱_袁红霞": SHEZHEN_VQA_SYSTEM_PROMPT_XIALUO,
}

# 使用固定题库的书籍 → 对应的 JSON Schema（跳过 _postprocess_qa_pairs）
BOOK_FIXED_QUESTION_BANK: dict[str, dict] = {
    "舌下络脉诊法图谱_袁红霞": XIALUO_QA_JSON_SCHEMA,
    "舌诊快速入门": KUAISU_QA_JSON_SCHEMA,
    "望舌诊病": WANGSHE_QA_JSON_SCHEMA,
    "舌诊全息论": QUANXI_QA_JSON_SCHEMA,
    "舌诊学": SHEZHENXUE_QA_JSON_SCHEMA,
    "舌诊十讲": SHIZJIANG_QA_JSON_SCHEMA,
}

# QA 输出中需要将具体图号（图7、图10…）替换为"图中"的书籍
BOOK_STRIP_FIGURE_NUMBERS: set[str] = {"舌诊十讲"}

# 书籍专用章节提取器（默认使用 _bianzhen_chapter）
BOOK_CHAPTER_EXTRACTOR: dict[str, object] = {
    "舌诊十讲": _shizjiang_lecture,
}

# 需要样本级 prompt 路由的书名 → {章节号: system_prompt}
BOOK_PER_SAMPLE_ROUTING: dict[str, dict[int, str]] = {
    "舌诊辨证图谱_周幸来": {
        2: SHEZHEN_VQA_SYSTEM_PROMPT_BIANZHEN_CH2,
        3: SHEZHEN_VQA_SYSTEM_PROMPT_BIANZHEN_CH3,
    },
    "舌诊快速入门": {
        2: SHEZHEN_VQA_SYSTEM_PROMPT_KUAISU_CH2,
        3: SHEZHEN_VQA_SYSTEM_PROMPT_KUAISU_CH3,
    },
    "望舌诊病": {
        2: SHEZHEN_VQA_SYSTEM_PROMPT_WANGSHE_CH2,
        3: SHEZHEN_VQA_SYSTEM_PROMPT_WANGSHE_CH3,
    },
    "舌诊全息论": {
        1: SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH1,
        3: SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH3,
        4: SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH4,
        5: SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH5,
        6: SHEZHEN_VQA_SYSTEM_PROMPT_QUANXI_CH6,
    },
    "舌诊学": {
        1: SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH1,
        4: SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH4,
        5: SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH5,
        6: SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH6,
        9: SHEZHEN_VQA_SYSTEM_PROMPT_SHEZHENXUE_CH9,
    },
    "舌诊十讲": {
        1: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L1,
        2: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L2,
        3: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L3,
        4: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L4,
        5: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L5,
        6: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L6,
        7: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L7,
        8: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L8,
        9: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L9,
        10: SHEZHEN_VQA_SYSTEM_PROMPT_SHIZJIANG_L10,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "shezhen_vqa_workdir"
DEFAULT_RERUN_DIR = WORK_DIR / "output" / "rerun_20260402" / "rerun_20260402"


class SplitRerunVQAProcessor(ShezhenVQAProcessor):
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        model_name: str,
        max_workers: int,
        temperature: float,
        mode: str,
        drop_noise_sections: bool = False,
        max_context_chars: int = 1500,
    ):
        self.input_root = input_root
        super().__init__(
            model_name=model_name,
            max_workers=max_workers,
            temperature=temperature,
            mode=mode,
            drop_noise_sections=drop_noise_sections,
            max_context_chars=max_context_chars,
            output_dir=output_root,
        )

    def _load_samples(self, book_name: str, limit: int | None, sample_rate: float, seed: int) -> list[dict]:
        sample_file = self.input_root / book_name / "split_samples.jsonl"
        if not sample_file.exists():
            raise FileNotFoundError(f"未找到样本文件: {sample_file}")

        import random

        rng = random.Random(seed)
        items: list[dict] = []
        skipped_noise_sections = 0
        with sample_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.drop_noise_sections:
                    _, is_noise = self._normalize_section_title(
                        str(row.get("section_title", "") or ""),
                        str(row.get("image_caption", "") or ""),
                    )
                    if is_noise:
                        skipped_noise_sections += 1
                        continue
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue
                items.append(row)
                if limit and len(items) >= limit:
                    break
        if self.drop_noise_sections:
            print(f"[过滤] {book_name}: 跳过噪声章节样本 {skipped_noise_sections} 条")
        return items

    def process_book(
        self,
        book_name: str,
        limit: int | None,
        sample_rate: float,
        seed: int,
    ) -> Path:
        samples = self._load_samples(book_name, limit=limit, sample_rate=sample_rate, seed=seed)
        if not samples:
            raise ValueError(f"{book_name} 没有可处理样本")

        user_inputs = [self._build_user_prompt(s) for s in samples]
        print(f"[加载] {book_name}: 样本 {len(samples)} 条")

        if book_name in BOOK_PER_SAMPLE_ROUTING:
            # 样本级 prompt 路由：每条样本根据章节号选择 system_prompt
            chapter_prompts = BOOK_PER_SAMPLE_ROUTING[book_name]

            extractor = BOOK_CHAPTER_EXTRACTOR.get(book_name, _bianzhen_chapter)

            def _per_sample_prompt(sample: dict, _ext=extractor) -> str:
                ch = _ext(str(sample.get("image_caption", "") or ""))
                return chapter_prompts.get(ch, next(iter(chapter_prompts.values()), ""))

            conversations = [
                [
                    {"role": "system", "content": _per_sample_prompt(s)},
                    {"role": "user", "content": u},
                ]
                for s, u in zip(samples, user_inputs)
            ]
            responses = self.llm.generate_from_conversations(conversations)
        else:
            system_prompt = BOOK_SPECIFIC_PROMPTS.get(book_name) or (
                SHEZHEN_VQA_SYSTEM_PROMPT_STRICT
                if self.mode == "strict"
                else SHEZHEN_VQA_SYSTEM_PROMPT_ENRICHED
            )
            json_schema = BOOK_FIXED_QUESTION_BANK.get(book_name, QA_JSON_SCHEMA)
            responses = self.llm.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
                json_schema=json_schema,
            )

        book_dir = self.output_dir / book_name
        book_dir.mkdir(parents=True, exist_ok=True)
        out_file = book_dir / "vqa_dataset.jsonl"
        debug_file = book_dir / "raw_debug.txt"
        written = 0
        empty_after_clean = 0
        no_qa = 0

        from tqdm import tqdm

        with debug_file.open("w", encoding="utf-8") as dbg, out_file.open("w", encoding="utf-8") as out:
            for idx, (sample, resp) in enumerate(
                tqdm(zip(samples, responses), total=len(samples), desc=f"解析 {book_name}")
            ):
                raw = resp or ""
                cleaned = self._clean_response(raw)

                dbg.write(f"# INDEX {idx}\n")
                dbg.write("# RAW\n")
                dbg.write(raw + "\n")
                dbg.write("# CLEANED\n")
                dbg.write(cleaned + "\n\n")

                if idx < 2:
                    preview = raw[:180].replace("\n", " ")
                    print(f"[调试] #{idx + 1} caption={self._caption_for_prompt(sample)[:30]}… resp={preview}")

                if not cleaned:
                    empty_after_clean += 1
                    continue

                qa_pairs = self._extract_qa_pairs(cleaned)
                if not qa_pairs:
                    no_qa += 1
                    continue
                is_fixed_bank = book_name in BOOK_FIXED_QUESTION_BANK
                if not is_fixed_bank:
                    qa_pairs = self._postprocess_qa_pairs(sample, qa_pairs)

                section_title_raw = str(sample.get("section_title", "") or "")
                section_title_clean, section_noise = self._normalize_section_title(
                    section_title_raw,
                    str(sample.get("image_caption", "") or ""),
                )
                strip_fig = book_name in BOOK_STRIP_FIGURE_NUMBERS
                for qa in qa_pairs:
                    q_text = qa["question"]
                    a_text = qa["answer"]
                    if strip_fig:
                        q_text = re.sub(r"图\d+中?", "图中", q_text)
                        a_text = re.sub(r"图\d+中?", "图中", a_text)
                        a_text = re.sub(r"(原文锚定句中?|根据原文|原文[讲中将]|据原文)", "", a_text)
                    record = {
                        "book_name": sample.get("book_name", ""),
                        "section_title": section_title_clean,
                        "section_title_raw": section_title_raw,
                        "section_title_is_noise": section_noise,
                        "image_path": sample.get("image_path", ""),
                        "image_caption": self._caption_for_prompt(sample),
                        "context_text": sample.get("context_text", ""),
                        "question": q_text,
                        "answer": a_text,
                        "generation_mode": self.mode,
                    }
                    if qa.get("type"):
                        record["type"] = qa["type"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        print(
            f"[完成] {book_name}: 写入 {written} 条 QA -> {out_file} "
            f"(清洗后为空 {empty_after_clean}, 无 qa_pair {no_qa})"
        )
        print(f"[调试] 原始响应日志: {debug_file}")
        return out_file


def _discover_books(input_root: Path) -> list[str]:
    return sorted(
        p.name for p in input_root.iterdir()
        if p.is_dir() and (p / "split_samples.jsonl").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从 rerun split_samples 生成舌诊 VQA")
    parser.add_argument("--book", type=str, help="只处理指定书名")
    parser.add_argument("--all", action="store_true", help="处理输入目录下全部书籍")
    parser.add_argument("--input-root", type=str, default=str(DEFAULT_RERUN_DIR), help="每本书下包含 split_samples.jsonl 的根目录")
    parser.add_argument("--output-root", type=str, default="", help="输出根目录，默认与 input-root 相同")
    parser.add_argument("--limit", type=int, default=None, help="每本书最多取多少条样本（调试用）")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="按比例随机抽样 (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    parser.add_argument("--mode", choices=["strict", "enriched"], default="enriched", help="提示词模式")
    parser.add_argument("--drop-noise-sections", action="store_true", help="丢弃疑似目录/版权等噪声章节")
    parser.add_argument("--max-context-chars", type=int, default=1500, help="单样本最大上下文字符数")
    parser.add_argument("--max-workers", type=int, default=5, help="LLM 并发数")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度")
    parser.add_argument("--model-name", type=str, default="MiniMax-M1", help="模型名")
    args = parser.parse_args()

    if not args.book and not args.all:
        parser.print_help()
        return

    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser() if args.output_root.strip() else input_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.book:
        targets = [args.book]
    else:
        targets = _discover_books(input_root)

    processor = SplitRerunVQAProcessor(
        input_root=input_root,
        output_root=output_root,
        model_name=args.model_name,
        max_workers=args.max_workers,
        temperature=args.temperature,
        mode=args.mode,
        drop_noise_sections=args.drop_noise_sections,
        max_context_chars=args.max_context_chars,
    )

    for book in targets:
        try:
            processor.process_book(
                book_name=book,
                limit=args.limit,
                sample_rate=args.sample_rate,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[异常] {book}: {e}")


if __name__ == "__main__":
    main()
