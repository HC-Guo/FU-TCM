# 数据目录说明

本目录包含全部最终数据产出，共 **113,065 条** QA 对，覆盖 30 本中医书籍。

## 目录结构

```
data/
├── maizhen_vqa/          # 望诊/脉诊 VQA — 5 本书, 3,386 条, 92 MB
├── shezhen_vqa/          # 舌诊 VQA    — 8 本书, 6,322 条, 253 MB
├── zhongyao_caotu_vqa/   # 中药彩图 VQA — 2 册,  10,036 条, 157 MB
└── medical_books_qa/     # 医学教材 QA  — 15 本, 93,321 条, 224 MB
```

---

## 1. maizhen_vqa — 望诊/脉诊 VQA

来源：面诊、脉诊、望诊类带图书籍，每张图生成多条 VQA。

| 书名 | 图片数 | 样本数 | VQA 条数 |
|------|--------|--------|----------|
| 中医望诊彩色图谱 | 206 | 206 | 386 |
| 中医脉诊临床图解 | 163 | 163 | 456 |
| 中西医结合望诊启迪 | 275 | 275 | 452 |
| 望诊之钥 十字面形诊治法 | 72 | 72 | 909 |
| 望面诊病图解 | 330 | 330 | 1,183 |
| **合计** | **1,046** | **1,046** | **3,386** |

每本书目录结构：
```
<书名>/
├── images/              # 从 PDF 提取的原始图片（.jpg）
├── split_samples.jsonl  # 切分后的图文样本（生成输入）
└── vqa_dataset.jsonl    # 最终 VQA 数据
```

**字段说明（vqa_dataset.jsonl）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `book_name` | string | 书名 |
| `section_title` | string | 章节标题 |
| `image_path` | string | 图片相对路径 |
| `image_caption` | string | 图片标注/说明 |
| `context_text` | string | 图片周围的上下文文本 |
| `question` | string | 问题 |
| `answer` | string | 答案 |
| `qa_type` | string | 题型（见下方题型表） |
| `generation_mode` | string | 生成模式（strict / enriched） |

**VQA 题型**：

| qa_type | 含义 |
|---------|------|
| `visual_feature` | 图像视觉特征识别 |
| `visual_recognition` | 视觉识别 |
| `visual_grounding` | 视觉定位 |
| `visual_mapping` | 视觉映射 |
| `diagnostic_mapping` | 诊断映射 |
| `clinical_reasoning` | 临床推理 |
| `clinical_analysis` | 临床分析 |
| `clinical_advice` | 临床建议 |
| `western_disease` | 现代医学疾病对应 |
| `tcm_pathogenesis` | 中医病机 |
| `treatment_prescription` | 治法方药 |
| `text_grounded_explanation` | 基于文本的解释 |
| `comprehensive_application` | 综合应用 |
| `spatial_location` | 空间定位 |
| `pulse_type_identification` | 脉型识别 |
| `pulse_waveform_feature` | 脉波特征 |
| `clinical_significance` | 临床意义 |

---

## 2. shezhen_vqa — 舌诊 VQA

来源：8 本舌诊带图书籍，每本有按章节定制的 prompt 和题型。

| 书名 | 图片数 | 样本数 | VQA 条数 |
|------|--------|--------|----------|
| 实用中医舌诊彩色图谱 | 136 | 135 | 782 |
| 望舌诊病 | 81 | 81 | 242 |
| 舌下络脉诊法图谱（袁红霞） | 197 | 190 | 417 |
| 舌诊全息论 | 330 | 159 | 628 |
| 舌诊十讲 | 205 | 183 | 219 |
| 舌诊学 | 2,175 | 1,115 | 1,716 |
| 舌诊快速入门 | 196 | 196 | 738 |
| 舌诊辨证图谱（周幸来） | 757 | 746 | 1,580 |
| **合计** | **4,077** | **2,805** | **6,322** |

目录结构与字段格式同 maizhen_vqa。数据位于 `rerun_20260402/` 子目录下。

---

## 3. zhongyao_caotu_vqa — 中药彩图 VQA

来源：《常用中药彩色图谱》上下册，对每味中药的彩色图片生成多维度问答。

| 册 | 图片数 | 样本数 | VQA 条数 |
|----|--------|--------|----------|
| 上册 | 876 | 859 | 5,272 |
| 下册 | 805 | 795 | 4,764 |
| **合计** | **1,681** | **1,654** | **10,036** |

**字段说明（vqa_dataset.jsonl）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `book_name` | string | 书名（中国中药草图谱） |
| `volume` | string | 上册 / 下册 |
| `drug_id` | int | 药材序号 |
| `title` | string | 药材名称（如"一年蓬"） |
| `image_path` | string | 图片相对路径 |
| `dimension` | string | 问答维度（外观识别 / 植物分类 / 功效主治 等） |
| `question` | string | 问题 |
| `answer` | string | 答案 |

---

## 4. medical_books_qa — 医学教材文本 QA

来源：15 本中医教材和药典 PDF，经 PDF→Markdown→分块→LLM 生成的纯文本 QA。

汇总文件：**all_books_qa_final.jsonl**（93,321 条）

覆盖书籍：

| 类别 | 书名 |
|------|------|
| 中医临床 | 中医内科学、中医外科学、中医妇科学、中医儿科学、中西医结合外科学 |
| 中医基础 | 中医基础理论、中医诊断学 |
| 中药方剂 | 中药学、中药炮制学、方剂学 |
| 现代药理 | 药理学 |
| 药典 | 药典一部（中药）、二部（化学药）、三部（生物制品）、四部（通则） |

**字段说明（all_books_qa_final.jsonl）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | string | 原始 PDF 路径 |
| `book_name` | string | 书名 |
| `raw_chunk` | string | 原始文本块 |
| `chunk_id` | int | 块序号 |
| `generated_content` | string | LLM 原始输出（含 think 标签） |

> 注：该文件保留了完整的生成记录（含 LLM 思考过程），可通过解析 `generated_content` 中的 `qa_pairs` JSON 提取最终 QA 对。
