# 舌诊辨证图谱_周幸来 VQA 提示词工程

## 章节路由逻辑

本书分为两大部分，结构差异极大，必须使用不同的题库引擎处理：

### 第二章路由条件
`section_title` 包含以下关键词之一：
- `舌色辨证`
- `舌形辨证`
- `舌态辨证`
- `苔质辨证`
- `苔色辨证`
- `舌脉辨证`
- `舌纹`
- `其他病变`

→ 使用 [`ch2_vqa_prompt.md`](./ch2_vqa_prompt.md)（10维基础理论题库）

### 第三章路由条件
`section_title` 包含 `【舌诊辨证】` 或 `第三章` 或 `临床`

→ 使用 [`ch3_vqa_prompt.md`](./ch3_vqa_prompt.md)（6维临床辨证题库）

## 数据特征对比

| 维度 | 第二章 | 第三章 |
|------|--------|--------|
| context_text 长度 | 长（200-600字） | 极短（10-30字） |
| 文本结构 | 理论阐述 + 西医机制 | 公式：舌质X + 苔Y = 证型Z |
| 图注格式 | 图2-X-X | 图3-X-X |
| 是否有治疗方案 | 偶有 | 有（中医简易疗法段落） |
| 是否有病程分期 | 偶有 | 有（●急性期 / ●慢性期） |

## 输出字段规范

每条 VQA 输出 JSON 格式：
```json
{
  "question": "问题文本",
  "answer": "答案文本（严格来自原文）",
  "type": "维度类型标识"
}
```

`type` 枚举值：
- 第二章：`moisture`、`color`、`shape_texture`、`mobility`、`core_pathogenesis`、`spatial_mapping`、`interaction_evolution`、`associated_symptoms`、`prognosis`、`western_mechanism`
- 第三章：`clinical_stage`、`visual_feature`（舌质/舌苔均用此值）、`syndrome_differentiation`、`specific_indicator`、`treatment`
