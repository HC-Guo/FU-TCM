# 中医多模态数据集生成系统

基于 [DataFlow](https://github.com/OpenDCAI/DataFlow) 框架，从中医带图书籍和教材 PDF 中自动生成 VQA（视觉问答）和 QA（文本问答）训练数据。

## 数据规模

| 数据集 | 书籍数 | QA 条数 | 说明 |
|--------|--------|---------|------|
| 望诊/脉诊 VQA | 5 本 | 3,386 | 面诊、脉诊、望诊类图文问答 |
| 舌诊 VQA | 8 本 | 6,322 | 舌象识别与辨证问答 |
| 中药彩图 VQA | 2 册 | 10,036 | 中药饮片形态辨识问答 |
| 医学教材 QA | 15 本 | 93,321 | 中医教材 + 药典文本问答 |
| **合计** | **30 本** | **113,065** | 总计约 920 MB |

## 项目结构

```
├── scripts/                        # 生成脚本
│   ├── vqa/                        # VQA 生成（15 个脚本）
│   │   ├── gen_shezhen_vqa.py              # 舌诊 VQA 主流程（PDF→切分→生成）
│   │   ├── gen_shezhen_vqa_final.py        # 舌诊基类 ShezhenVQAProcessor
│   │   ├── gen_shezhen_vqa_from_split.py   # 舌诊 VQA（从 split_samples 生成，含8本书专用 prompt）
│   │   ├── gen_bianzhen_vqa.py             # 辨证类 VQA
│   │   ├── gen_shitu_vqa.py                # 示图类 VQA
│   │   ├── gen_zhongxi_ch{1,3,5-10}_vqa.py # 《中西医结合望诊启迪》各章 VQA（8个）
│   │   ├── gen_zhongyao_caotu_vqa.py       # 中药彩图 VQA
│   │   └── regenerate_unified_book_vqa.py  # 按书名重跑 VQA 工具
│   ├── qa/                         # 文本 QA 生成（3 个脚本）
│   │   ├── medical_pdf_to_qa_pipeline.py   # 主流程：PDF→Markdown→分块→QA 生成
│   │   ├── gen_ypd2_qa.py                  # 药典二部专用
│   │   └── gen_ypd34_qa.py                 # 药典三部、四部专用
│   └── chunkers/                   # 自定义文本分块器（8 个）
│       ├── config.py                       # 分块配置
│       └── custom_chunker_*.py             # 各书专用分块逻辑
│
├── data/                           # 最终数据产出
│   ├── maizhen_vqa/                # 望诊/脉诊 VQA（5 本书）
│   ├── shezhen_vqa/                # 舌诊 VQA（8 本书）
│   ├── zhongyao_caotu_vqa/         # 中药彩图 VQA（上下册）
│   └── medical_books_qa/           # 医学教材 QA（15 本书）
│       └── all_books_qa_final.jsonl    # 全量汇总
│
├── prompt_engineering/             # 提示词设计文档
│   ├── common_design_rules.md          # 通用设计原则
│   ├── shezhen/                        # 舌诊 prompt
│   ├── mianzhen/                       # 面诊 prompt
│   ├── maizhen/                        # 脉诊 prompt
│   ├── wangzhen_mixed/                 # 望诊综合 prompt
│   └── zhongyao_caotu/                 # 中药彩图 prompt
│
└── framework/                      # DataFlow 开源框架
    └── DataFlow-main/
```

## 数据格式

### VQA 输出（vqa_dataset.jsonl）

```json
{
  "book_name": "舌诊辨证图谱_周幸来",
  "section_title": "第二章 常见舌象",
  "image_path": "images/fig2-1.jpg",
  "image_caption": "图2-1 淡红舌薄白苔",
  "context_text": "淡红舌薄白苔为正常舌象...",
  "question": "图中舌体的颜色和舌苔的形态分别是什么？",
  "answer": "舌体呈淡红色，舌苔薄白，均匀覆盖舌面。",
  "qa_type": "visual_feature",
  "generation_mode": "enriched"
}
```

### QA 输出（all_books_qa_final.jsonl）

```json
{
  "id": "qa_00001",
  "source": "中医内科学",
  "question": "感冒的风寒证和风热证在临床表现上有哪些主要区别？",
  "answer": "风寒证以恶寒重、发热轻、无汗、鼻塞流清涕...",
  "evidence": "原文摘录..."
}
```

## VQA 生成模式

项目针对不同书籍结构设计了多种生成模式：

| 模式 | 适用场景 | 脚本 |
|------|---------|------|
| **理论图解模式** | 单图配理论说明（ch1, ch3） | `gen_zhongxi_ch1/ch3_vqa.py` |
| **雷达扫描裂变模式** | 病例图 × N 个命中部位，每部位独立 5 题 CoT 题组（ch5-10） | `gen_zhongxi_ch5~10_vqa.py` |
| **章节路由模式** | 同一本书不同章节使用不同 prompt | `gen_shezhen_vqa_from_split.py` |
| **固定题库模式** | 预定义题型结构，适合格式统一的书籍 | 同上（舌诊快速入门等） |
| **通用 enriched 模式** | 图文对生成开放式 QA | `gen_shezhen_vqa_final.py` |

## 环境配置

### 依赖

```bash
pip install open-dataflow pandas tqdm
```

### 环境变量

```bash
export DF_API_KEY="your-api-key"
export MINIMAX_API_URL="https://api.minimax.chat/v1/chat/completions"
export MINIMAX_MODEL="MiniMax-M1"
```

## 使用示例

```bash
# 舌诊 VQA：处理全部书籍
python scripts/vqa/gen_shezhen_vqa_from_split.py --all --input-root data/shezhen_vqa/rerun_20260402

# 舌诊 VQA：处理单本书
python scripts/vqa/gen_shezhen_vqa_from_split.py --book 舌诊辨证图谱_周幸来

# 中西医望诊 VQA：第5章
python scripts/vqa/gen_zhongxi_ch5_vqa.py --input-dir data/maizhen_vqa/中西医结合望诊启迪

# 中药彩图 VQA
python scripts/vqa/gen_zhongyao_caotu_vqa.py

# 医学教材 QA（全流程：PDF→Markdown→分块→QA）
python scripts/qa/medical_pdf_to_qa_pipeline.py

# 药典二部 QA
python scripts/qa/gen_ypd2_qa.py
```

## 书籍清单

### VQA 书籍（望诊/脉诊/舌诊）

| 领域 | 书名 |
|------|------|
| 望诊 | 中医望诊彩色图谱、望面诊病图解、望诊之钥 十字面形诊治法 |
| 脉诊 | 中医脉诊临床图解 |
| 望诊(中西医) | 中西医结合望诊启迪（ch1/ch3/ch5-ch10） |
| 舌诊 | 实用中医舌诊彩色图谱、望舌诊病、舌下络脉诊法图谱、舌诊全息论、舌诊十讲、舌诊学、舌诊快速入门、舌诊辨证图谱 |
| 中药彩图 | 常用中药彩色图谱（上册、下册） |

### QA 教材（15 本）

中医内科学、中医基础理论、中医外科学、中医妇科学、中医儿科学、中医诊断学、中药学、中药炮制学、中西医结合外科学、方剂学、药理学、药典一部/二部/三部/四部
