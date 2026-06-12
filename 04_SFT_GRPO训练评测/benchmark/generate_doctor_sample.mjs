import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const benchDir = path.join(root, 'benchmark');
const seed = 20260610;

function mulberry32(a) {
  return function rng() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(seed);

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(benchDir, name), 'utf8'));
}

function difficultyLabel(level) {
  return level === 'hard' ? '疑难' : level === 'medium' ? '中等' : '简单';
}

function syndromeDifficulty(count) {
  if (count >= 3) return 'hard';
  if (count === 2) return 'medium';
  return 'simple';
}

function countBy(arr, keyFn) {
  const counts = new Map();
  for (const item of arr) {
    const key = keyFn(item);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].map(([key, count]) => ({ key, count }));
}

const textBench = readJson('tcm_benchmark_v4.json');
const bianzhengBench = readJson('bianzheng_mcq_test.json');
const visionBench = readJson('tcm_benchmark_vision_v4.json');

const bianFlat = Object.entries(bianzhengBench).flatMap(([category, rows]) =>
  rows.map((item, idx) => {
    const syndromeCount = Array.isArray(item.correct_syndromes) ? item.correct_syndromes.length : 1;
    return {
      source_file: 'bianzheng_mcq_test.json',
      bench: '辨证医案',
      record_id: item.case_id || `bianzheng_${category}_${idx + 1}`,
      category,
      task_type: '辨证分型 / 医案推理',
      modality: '纯文本',
      syndrome_count: syndromeCount,
      difficulty: syndromeDifficulty(syndromeCount),
      question: item.question,
      options: item.options,
      answer_idx: item.answer_idx,
      answer_text: item.answer_text,
      correct_syndromes: item.correct_syndromes || [],
    };
  })
);

function pickBianzheng() {
  const quotas = [
    { category: 'standard', syndrome_count: 1, n: 8 },
    { category: 'standard', syndrome_count: 2, n: 5 },
    { category: 'standard', syndrome_count: 3, n: 1 },
    { category: 'colloquial', syndrome_count: 1, n: 6 },
    { category: 'colloquial', syndrome_count: 2, n: 3 },
    { category: 'colloquial', syndrome_count: 3, n: 1 },
  ];
  const selected = [];
  const usedAnswers = new Set();
  for (const quota of quotas) {
    const pool = shuffle(
      bianFlat.filter(
        item => item.category === quota.category && item.syndrome_count === quota.syndrome_count
      )
    );
    const picked = [];
    for (const item of pool) {
      if (usedAnswers.has(item.answer_text)) continue;
      picked.push(item);
      usedAnswers.add(item.answer_text);
      if (picked.length === quota.n) break;
    }
    if (picked.length < quota.n) {
      for (const item of pool) {
        if (picked.includes(item)) continue;
        picked.push(item);
        if (picked.length === quota.n) break;
      }
    }
    selected.push(...picked);
  }
  return selected;
}

function formulaType(question) {
  if (/不包括|错误|加减|变化|哪组|特点|辨析|区别/.test(question)) return '相似方/加减辨析';
  if (/组成|由哪些药物|药物组成|含.*的|包括/.test(question)) return '方剂组成';
  if (/煎|服|用法|使用|每日|频率|制法|制作/.test(question)) return '煎服/用法';
  if (/治则|治法|立法|首选|选用|该方剂是/.test(question)) return '立法处方/治则';
  return '主治功效/应用';
}

function herbType(question) {
  if (/性味|归经/.test(question)) return '性味归经';
  if (/配伍|佐|得|为使|使药/.test(question)) return '配伍';
  if (/炮制|炒|炙|去心|去毛|制法|洗/.test(question)) return '炮制';
  if (/《|记载|产地|别名|质地|颜色|哪种中药|古代/.test(question)) return '冷门药/文献记载';
  return '功效主治';
}

function textDifficulty(item) {
  const question = item.question || '';
  if (/不包括|错误|加减|相似|哪组|文献|《|冷门|古代|产地/.test(question) || question.length > 80) {
    return 'hard';
  }
  if (/治疗|主治|病机|治则|治法|配伍|首选|作用/.test(question) || question.length > 45) {
    return 'medium';
  }
  return 'simple';
}

const textFlat = Object.entries(textBench).flatMap(([section, rows]) =>
  rows.map((item, idx) => {
    const subtype = section === '药方' ? formulaType(item.question) : herbType(item.question);
    const answerIdx = item.answer_idx || item.answer;
    return {
      source_file: 'tcm_benchmark_v4.json',
      bench: '方药知识',
      record_id: `text_${section}_${idx + 1}`,
      category: section,
      task_type: section === '药方' ? `立法处方 / ${subtype}` : `知识问答 / ${subtype}`,
      subtype,
      modality: '纯文本',
      difficulty: textDifficulty(item),
      question: item.question,
      options: item.options,
      answer_idx: answerIdx,
      answer_text: item.options?.[answerIdx] || item.answer,
      original_type: item.original_type,
      original_question: item.original_question,
    };
  })
);

function pickBySubtype(section, subtypeQuotas) {
  const selected = [];
  const usedQuestions = new Set();
  for (const [subtype, quota] of Object.entries(subtypeQuotas)) {
    const pool = shuffle(textFlat.filter(item => item.category === section && item.subtype === subtype));
    for (const item of pool) {
      if (usedQuestions.has(item.question)) continue;
      selected.push(item);
      usedQuestions.add(item.question);
      if (selected.filter(row => row.subtype === subtype).length === quota) break;
    }
  }
  return selected;
}

function sourceSeries(imagePath) {
  const base = path.basename(imagePath || '');
  return base.replace(/_[0-9a-f]{64}\.jpg$/i, '').replace(/\.jpg$/i, '');
}

function visionTaskType(question) {
  if (/治则|治法|病机|病症|病证|临床意义|方剂|西医诊断|主治|提示/.test(question)) {
    return '图像 + 病证/治法综合推理';
  }
  return '舌象/面象/脉象识别';
}

function visionDifficulty(question, answer) {
  if (/不包括|首选|方剂|西医诊断|心理特征|病程|综合/.test(question) || String(answer || '').length > 35) {
    return 'hard';
  }
  if (/提示|病机|病症|证|治则|治法|临床意义/.test(question)) return 'medium';
  return 'simple';
}

const imageDirs = [
  path.join(root, 'sft_merged/images'),
  path.join(root, 'sft_image_data/unified/images'),
  path.join(root, 'sft_image_data/unified_remaining/images'),
  path.join(root, 'sft_image_data/unified_sampled_500/images'),
];

function resolveImage(imagePath) {
  const base = path.basename(imagePath || '');
  for (const dir of imageDirs) {
    const candidate = path.join(dir, base);
    if (fs.existsSync(candidate)) return candidate;
  }
  return '';
}

const visionFlat = visionBench.map((item, idx) => {
  const imagePath = item.image?.[0] || '';
  const taskType = visionTaskType(item.question || '');
  return {
    source_file: 'tcm_benchmark_vision_v4.json',
    bench: '视觉图文',
    record_id: `vision_${idx + 1}`,
    category: item.category || '其他',
    task_type: taskType,
    modality: taskType.startsWith('图像 +') ? '多模态综合' : '图文',
    difficulty: visionDifficulty(item.question || '', item.answer || ''),
    question: item.question,
    options: item.options,
    answer_idx: item.answer_idx,
    answer_text: item.answer,
    image_path: imagePath,
    image_abs_path: resolveImage(imagePath),
    source_series: sourceSeries(imagePath),
  };
});

function pickVision() {
  const quotas = [
    { category: '舌诊', task_type: '舌象/面象/脉象识别', n: 4 },
    { category: '舌诊', task_type: '图像 + 病证/治法综合推理', n: 3 },
    { category: '面诊', task_type: '舌象/面象/脉象识别', n: 2 },
    { category: '面诊', task_type: '图像 + 病证/治法综合推理', n: 1 },
    { category: '脉诊', task_type: '舌象/面象/脉象识别', n: 3 },
    { category: '脉诊', task_type: '图像 + 病证/治法综合推理', n: 1 },
    { category: '其他', task_type: '舌象/面象/脉象识别', n: 1 },
    { category: '其他', task_type: '图像 + 病证/治法综合推理', n: 1 },
  ];
  const selected = [];
  const used = new Set();
  const seriesCounts = new Map();
  for (const quota of quotas) {
    const pool = shuffle(visionFlat.filter(item => item.category === quota.category && item.task_type === quota.task_type));
    let count = 0;
    for (const item of pool) {
      if (used.has(item.record_id)) continue;
      const current = seriesCounts.get(item.source_series) || 0;
      if (current >= 3) continue;
      selected.push(item);
      used.add(item.record_id);
      seriesCounts.set(item.source_series, current + 1);
      count += 1;
      if (count === quota.n) break;
    }
    if (count < quota.n) {
      const fallback = shuffle(visionFlat.filter(item => item.category === quota.category));
      for (const item of fallback) {
        if (used.has(item.record_id)) continue;
        const current = seriesCounts.get(item.source_series) || 0;
        if (current >= 3) continue;
        selected.push(item);
        used.add(item.record_id);
        seriesCounts.set(item.source_series, current + 1);
        count += 1;
        if (count === quota.n) break;
      }
    }
  }
  return selected;
}

const bianSelected = pickBianzheng();
const textSelected = [
  ...pickBySubtype('药方', {
    方剂组成: 2,
    '主治功效/应用': 2,
    '煎服/用法': 2,
    '立法处方/治则': 2,
    '相似方/加减辨析': 2,
  }),
  ...pickBySubtype('中药', {
    性味归经: 2,
    功效主治: 2,
    炮制: 2,
    配伍: 2,
    '冷门药/文献记载': 2,
  }),
];

const forcedTextReplacements = new Map([
  ['text_药方_145', 'text_药方_184'],
  ['text_药方_283', 'text_药方_59'],
]);

for (let idx = 0; idx < textSelected.length; idx += 1) {
  const replacementId = forcedTextReplacements.get(textSelected[idx].record_id);
  if (!replacementId) continue;
  const replacement = textFlat.find(item => item.record_id === replacementId);
  const alreadySelected = textSelected.some((item, currentIdx) => currentIdx !== idx && item.record_id === replacementId);
  if (replacement && !alreadySelected) textSelected[idx] = replacement;
}
const visionSelected = pickVision();

const allSelected = [...bianSelected, ...textSelected, ...visionSelected].map((item, idx) => ({
  sample_no: idx + 1,
  difficulty_label: difficultyLabel(item.difficulty),
  ...item,
}));

const summary = {
  seed,
  total: allSelected.length,
  by_bench: countBy(allSelected, item => item.bench),
  by_difficulty: countBy(allSelected, item => item.difficulty_label),
  by_modality: countBy(allSelected, item => item.modality),
  bianzheng_count: bianSelected.length,
  bianzheng_answer_text_unique: new Set(bianSelected.map(item => item.answer_text)).size,
  bianzheng_syndrome_count: countBy(bianSelected, item => `${item.syndrome_count}证型`),
  bianzheng_category: countBy(bianSelected, item => item.category),
  vision_category: countBy(visionSelected, item => item.category),
  vision_source_series_count: new Set(visionSelected.map(item => item.source_series)).size,
  vision_missing_images: visionSelected.filter(item => !item.image_abs_path).length,
};

function statRows(rows) {
  return rows.map(row => `<tr><td>${esc(row.key)}</td><td>${row.count}</td></tr>`).join('');
}

function optionsHtml(options, answerIdx) {
  if (!options) return '';
  return `<ol class="options">${Object.entries(options).map(([key, value]) => {
    const correctClass = key === answerIdx ? ' correct' : '';
    return `<li class="${correctClass}"><span class="opt">${esc(key)}</span>${esc(value)}</li>`;
  }).join('')}</ol>`;
}

function itemHtml(item) {
  const tags = [item.bench, item.category, item.task_type, item.difficulty_label, item.modality];
  if (item.syndrome_count) tags.push(`${item.syndrome_count}证型`);
  if (item.source_series) tags.push(`来源：${item.source_series}`);
  const imageHtml = item.image_abs_path
    ? `<img src="${esc(item.image_abs_path)}" alt="${esc(item.source_series || 'vision image')}">`
    : item.image_path
      ? `<p class="image-missing">图片待映射：<code>${esc(item.image_path)}</code></p>`
      : '';
  const syndromes = item.correct_syndromes?.length
    ? `<p class="small"><strong>证型列表：</strong>${esc(item.correct_syndromes.join('、'))}</p>`
    : '';
  const original = item.original_question
    ? `<p class="small"><strong>原始问题：</strong>${esc(item.original_question)}</p>`
    : '';
  return `
    <div class="question-card">
      <div class="q-head">
        <div class="q-no">#${item.sample_no}</div>
        <div>${tags.map(tag => `<span class="tag">${esc(tag)}</span>`).join('')}</div>
      </div>
      ${imageHtml}
      <p class="question-text">${esc(item.question).replaceAll('\n', '<br>')}</p>
      ${optionsHtml(item.options, item.answer_idx)}
      <div class="answer">答案：<strong>${esc(item.answer_idx || '')}</strong> ${esc(item.answer_text || '')}</div>
      ${syndromes}
      ${original}
      <p class="small"><strong>ID：</strong>${esc(item.record_id)}　<strong>来源文件：</strong><code>${esc(item.source_file)}</code></p>
    </div>`;
}

function sectionHtml(title, items) {
  return `<h2>${esc(title)}</h2>${items.map(itemHtml).join('\n')}`;
}

const byId = new Map(allSelected.map(item => [item.record_id, item]));
const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>医生测试分层抽样样例包 20260610</title>
  <style>
    :root { --bg:#f7f4ee; --paper:#fffdf9; --ink:#1f241f; --muted:#636b60; --line:#ded5c8; --accent:#7b3f2c; --accent2:#2f6756; --soft:#efe7db; --ok:#ecf7ee; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.62; }
    .page { max-width:1120px; margin:0 auto; padding:38px 22px 64px; }
    .hero { border-bottom:2px solid var(--line); padding-bottom:18px; margin-bottom:26px; }
    h1 { margin:0 0 8px; font-size:30px; letter-spacing:0; }
    h2 { margin:34px 0 14px; font-size:22px; color:var(--accent); }
    p { margin:8px 0; }
    code { background:#eee4d7; border-radius:4px; padding:1px 5px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; }
    table { border-collapse:collapse; width:100%; background:var(--paper); border:1px solid var(--line); margin:12px 0 18px; font-size:14px; }
    th,td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
    th { background:var(--soft); }
    .summary-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:16px 0; }
    .metric { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:13px; }
    .num { font-size:28px; font-weight:800; color:var(--accent); }
    .label { color:var(--muted); font-size:13px; }
    .note { background:#fff0d4; border:1px solid #edd19a; border-radius:8px; padding:12px 14px; margin:14px 0; }
    .question-card { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:15px; margin:13px 0; break-inside:avoid; }
    .q-head { display:flex; gap:10px; align-items:flex-start; margin-bottom:8px; }
    .q-no { min-width:46px; color:var(--accent); font-weight:800; }
    .tag { display:inline-block; border:1px solid var(--line); background:#fbf6ed; border-radius:999px; padding:2px 8px; margin:2px 3px 2px 0; font-size:12px; color:var(--muted); }
    .question-text { white-space:normal; font-weight:600; }
    .options { margin:8px 0 10px; padding-left:24px; }
    .options li { margin:4px 0; padding:3px 6px; border-radius:6px; }
    .options li.correct { background:var(--ok); border:1px solid #b8dfc0; }
    .opt { display:inline-block; min-width:22px; font-weight:700; color:var(--accent2); }
    .answer { margin-top:8px; padding:8px 10px; border-left:4px solid var(--accent2); background:#f0f6f2; }
    .small { color:var(--muted); font-size:13px; }
    img { max-width:260px; max-height:220px; display:block; border:1px solid var(--line); border-radius:8px; margin:8px 0 10px; background:#fff; }
    .image-missing { color:#8b4a2f; }
    @media (max-width:780px) { .page{padding:26px 14px 48px;} .summary-grid{grid-template-columns:1fr 1fr;} .q-head{display:block;} table{font-size:13px;} }
    @media print { body{background:#fff;} .question-card{page-break-inside:avoid;} }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>医生测试分层抽样样例包</h1>
      <p class="small">随机种子：<code>${seed}</code>；生成日期：2026-06-10；来源目录：<code>${esc(benchDir)}</code></p>
      <div class="note"><strong>说明：</strong>这是审阅版，包含正确答案和分层标签，便于检查抽样是否合理。正式给医生作答前应导出盲测版，并重新平衡 A/B/C/D 选项位置。</div>
    </div>

    <h2>抽样概览</h2>
    <div class="summary-grid">
      <div class="metric"><div class="num">${summary.total}</div><div class="label">正式样例题数</div></div>
      <div class="metric"><div class="num">${summary.bianzheng_count}</div><div class="label">辨证医案题</div></div>
      <div class="metric"><div class="num">${summary.bianzheng_answer_text_unique}</div><div class="label">辨证答案组合数</div></div>
      <div class="metric"><div class="num">${summary.vision_source_series_count}</div><div class="label">视觉图片来源系列数</div></div>
    </div>
    <table>
      <thead><tr><th>Bench 分布</th><th>题数</th></tr></thead>
      <tbody>${statRows(summary.by_bench)}</tbody>
    </table>
    <table>
      <thead><tr><th>辨证类别</th><th>题数</th></tr></thead>
      <tbody>${statRows(summary.bianzheng_category)}</tbody>
    </table>
    <table>
      <thead><tr><th>辨证证型数量</th><th>题数</th></tr></thead>
      <tbody>${statRows(summary.bianzheng_syndrome_count)}</tbody>
    </table>
    <table>
      <thead><tr><th>视觉类别</th><th>题数</th></tr></thead>
      <tbody>${statRows(summary.vision_category)}</tbody>
    </table>

    ${sectionHtml('一、辨证医案：24 题（答案证型组合不重复）', bianSelected.map(item => byId.get(item.record_id)))}
    ${sectionHtml('二、方药知识：20 题', textSelected.map(item => byId.get(item.record_id)))}
    ${sectionHtml('三、视觉图文：16 题', visionSelected.map(item => byId.get(item.record_id)))}
  </div>
</body>
</html>`;

const outHtml = path.join(benchDir, 'doctor_sample_20260610_review.html');
const outJson = path.join(benchDir, 'doctor_sample_20260610_review.json');

fs.writeFileSync(outHtml, html, 'utf8');
fs.writeFileSync(outJson, JSON.stringify({ summary, items: allSelected }, null, 2), 'utf8');

console.log(JSON.stringify({ outHtml, outJson, summary }, null, 2));
