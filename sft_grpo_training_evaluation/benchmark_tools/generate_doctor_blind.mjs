import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const benchDir = path.join(root, 'benchmark');
const seed = 20260611;
const sourceJson = path.join(benchDir, 'doctor_sample_20260610_review.json');

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
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function countBy(arr, keyFn) {
  const counts = new Map();
  for (const item of arr) {
    const key = keyFn(item);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].map(([key, count]) => ({ key, count }));
}

const review = JSON.parse(fs.readFileSync(sourceJson, 'utf8'));
const items = review.items;
const letters = ['A', 'B', 'C', 'D'];
const targetLetters = shuffle(letters.flatMap(letter => Array(15).fill(letter)));

function relabelOptions(item, targetCorrectLetter) {
  const originalOptions = item.options || {};
  const originalCorrectText = originalOptions[item.answer_idx] ?? item.answer_text;
  const nonCorrect = Object.entries(originalOptions)
    .filter(([key]) => key !== item.answer_idx)
    .map(([old_idx, text]) => ({ old_idx, text }));
  const shuffledWrong = shuffle(nonCorrect);
  const remainingLetters = shuffle(letters.filter(letter => letter !== targetCorrectLetter));
  const newOptions = { [targetCorrectLetter]: originalCorrectText };
  const optionMap = [{
    new_idx: targetCorrectLetter,
    old_idx: item.answer_idx,
    text: originalCorrectText,
    is_correct: true,
  }];

  for (let i = 0; i < shuffledWrong.length; i += 1) {
    const newIdx = remainingLetters[i];
    newOptions[newIdx] = shuffledWrong[i].text;
    optionMap.push({
      new_idx: newIdx,
      old_idx: shuffledWrong[i].old_idx,
      text: shuffledWrong[i].text,
      is_correct: false,
    });
  }

  return {
    newOptions: Object.fromEntries(letters.map(letter => [letter, newOptions[letter]])),
    optionMap: optionMap.sort((a, b) => letters.indexOf(a.new_idx) - letters.indexOf(b.new_idx)),
  };
}

function cleanQuestionText(question) {
  const lines = String(question ?? '').split('\n');
  const optionStart = lines.findIndex((line, idx) => {
    const trimmed = line.trim();
    if (!/^A[.．、]\s*/.test(trimmed)) return false;
    const rest = lines.slice(idx + 1, idx + 4).map(nextLine => nextLine.trim());
    return /^B[.．、]\s*/.test(rest[0] || '')
      && /^C[.．、]\s*/.test(rest[1] || '')
      && /^D[.．、]\s*/.test(rest[2] || '');
  });
  if (optionStart === -1) return String(question ?? '');
  return lines.slice(0, optionStart).join('\n').trim();
}

const blindItems = [];
const answerKey = [];

for (let idx = 0; idx < items.length; idx += 1) {
  const item = items[idx];
  const targetCorrectLetter = targetLetters[idx];
  const { newOptions, optionMap } = relabelOptions(item, targetCorrectLetter);

  blindItems.push({
    sample_no: item.sample_no,
    question: cleanQuestionText(item.question),
    options: newOptions,
    image_abs_path: item.image_abs_path || '',
    image_path: item.image_path || '',
    modality: item.image_abs_path || item.image_path ? '图文题' : '文字题',
  });

  answerKey.push({
    sample_no: item.sample_no,
    record_id: item.record_id,
    bench: item.bench,
    category: item.category,
    subtype: item.subtype || '',
    task_type: item.task_type,
    difficulty_label: item.difficulty_label,
    original_answer_idx: item.answer_idx,
    blind_answer_idx: targetCorrectLetter,
    answer_text: item.answer_text,
    question: cleanQuestionText(item.question),
    option_map: optionMap,
  });
}

const summary = {
  seed,
  source: sourceJson,
  total: blindItems.length,
  blind_answer_distribution: countBy(answerKey, item => item.blind_answer_idx),
  by_modality: countBy(blindItems, item => item.modality),
};

function optionListHtml(options) {
  return `<ol class="options">${letters.map(letter =>
    `<li><span class="opt">${letter}</span>${esc(options[letter])}</li>`
  ).join('')}</ol>`;
}

function questionHtml(item) {
  const imageHtml = item.image_abs_path
    ? `<img src="${esc(item.image_abs_path)}" alt="题目图片">`
    : '';
  return `
    <div class="question-card">
      <div class="q-head">
        <div class="q-no">第 ${item.sample_no} 题</div>
        <span class="tag">${esc(item.modality)}</span>
      </div>
      ${imageHtml}
      <p class="question-text">${esc(item.question).replaceAll('\n', '<br>')}</p>
      ${optionListHtml(item.options)}
      <div class="response-line">
        <span>作答：□ A　□ B　□ C　□ D</span>
        <span>题目问题：□ 无　□ 有</span>
      </div>
    </div>`;
}

function statRows(rows) {
  return rows.map(row => `<tr><td>${esc(row.key)}</td><td>${row.count}</td></tr>`).join('');
}

const blindHtml = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>中医 Benchmark 医生盲测题本</title>
  <style>
    :root { --bg:#f7f4ee; --paper:#fffdf9; --ink:#1f241f; --muted:#636b60; --line:#ded5c8; --accent:#7b3f2c; --accent2:#2f6756; --soft:#efe7db; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.62; }
    .page { max-width:1080px; margin:0 auto; padding:36px 22px 64px; }
    .hero { border-bottom:2px solid var(--line); padding-bottom:18px; margin-bottom:24px; }
    h1 { margin:0 0 8px; font-size:30px; letter-spacing:0; }
    h2 { margin:30px 0 12px; font-size:21px; color:var(--accent); }
    p { margin:8px 0; }
    table { border-collapse:collapse; width:100%; background:var(--paper); border:1px solid var(--line); margin:12px 0 18px; font-size:14px; }
    th,td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
    th { background:var(--soft); }
    .small { color:var(--muted); font-size:13px; }
    .note { background:#fff0d4; border:1px solid #edd19a; border-radius:8px; padding:12px 14px; margin:14px 0; }
    .question-card { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:15px; margin:13px 0; break-inside:avoid; }
    .q-head { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
    .q-no { color:var(--accent); font-weight:800; min-width:76px; }
    .tag { display:inline-block; border:1px solid var(--line); background:#fbf6ed; border-radius:999px; padding:2px 8px; font-size:12px; color:var(--muted); }
    .question-text { font-weight:600; }
    .options { margin:8px 0 10px; padding-left:24px; }
    .options li { margin:4px 0; padding:3px 6px; border-radius:6px; }
    .opt { display:inline-block; min-width:22px; font-weight:700; color:var(--accent2); }
    .response-line { display:flex; flex-wrap:wrap; gap:18px; margin-top:12px; padding-top:10px; border-top:1px dashed var(--line); color:var(--muted); font-size:14px; }
    img { max-width:300px; max-height:240px; display:block; border:1px solid var(--line); border-radius:8px; margin:8px 0 10px; background:#fff; }
    @media (max-width:780px) { .page{padding:26px 14px 48px;} .q-head{display:block;} table{font-size:13px;} }
    @media print { body{background:#fff;} .question-card{page-break-inside:avoid;} .page{max-width:none;} }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>中医 Benchmark 医生盲测题本</h1>
    </div>

    <h2>题本概览</h2>
    <table>
      <thead><tr><th>题型</th><th>题数</th></tr></thead>
      <tbody>${statRows(summary.by_modality)}</tbody>
    </table>

    <h2>正式题目</h2>
    ${blindItems.map(questionHtml).join('\n')}
  </div>
</body>
</html>`;

const blindJson = {
  summary: {
    total: blindItems.length,
    seed,
    note: '盲测题本不包含正确答案；评分请使用 doctor_sample_20260610_answer_key.json。',
  },
  items: blindItems,
};

const outBlindHtml = path.join(benchDir, 'doctor_sample_20260610_blind.html');
const outBlindJson = path.join(benchDir, 'doctor_sample_20260610_blind.json');
const outAnswerKey = path.join(benchDir, 'doctor_sample_20260610_answer_key.json');

fs.writeFileSync(outBlindHtml, blindHtml, 'utf8');
fs.writeFileSync(outBlindJson, JSON.stringify(blindJson, null, 2), 'utf8');
fs.writeFileSync(outAnswerKey, JSON.stringify({ summary, answers: answerKey }, null, 2), 'utf8');

console.log(JSON.stringify({
  outBlindHtml,
  outBlindJson,
  outAnswerKey,
  summary,
}, null, 2));
