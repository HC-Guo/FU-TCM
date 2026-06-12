import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const benchDir = path.join(root, 'benchmark');
const sourcePath = path.join(benchDir, 'doctor_sample_20260610_answer_key.json');
const outPath = path.join(benchDir, 'doctor_sample_20260610_answer_key.html');

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function countBy(rows, keyFn) {
  const counts = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].map(([key, count]) => ({ key, count }));
}

const data = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
const answers = data.answers;
const dist = countBy(answers, item => item.blind_answer_idx).sort((a, b) => a.key.localeCompare(b.key));
const benchDist = countBy(answers, item => item.bench);

function statRows(rows) {
  return rows.map(row => `<tr><td>${esc(row.key)}</td><td>${row.count}</td></tr>`).join('');
}

function optionMapHtml(item) {
  return `<div class="options-map">${item.option_map.map(option => {
    const cls = option.is_correct ? ' correct' : '';
    return `<span class="opt-map${cls}">${esc(option.new_idx)}：${esc(option.text)}</span>`;
  }).join('')}</div>`;
}

function answerRow(item) {
  return `
    <tr>
      <td class="no">${item.sample_no}</td>
      <td class="answer">${esc(item.blind_answer_idx)}</td>
      <td>${esc(item.answer_text)}</td>
      <td>${esc(item.question).replaceAll('\n', '<br>')}</td>
      <td>
        <span class="tag">${esc(item.bench)}</span>
        <span class="tag">${esc(item.category)}</span>
        <span class="tag">${esc(item.difficulty_label)}</span>
        ${item.subtype ? `<span class="tag">${esc(item.subtype)}</span>` : ''}
        <p class="small"><strong>ID：</strong>${esc(item.record_id)}</p>
        ${optionMapHtml(item)}
      </td>
    </tr>`;
}

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>中医 Benchmark 医生盲测答案表</title>
  <style>
    :root { --bg:#f7f4ee; --paper:#fffdf9; --ink:#1f241f; --muted:#636b60; --line:#ded5c8; --accent:#7b3f2c; --accent2:#2f6756; --soft:#efe7db; --ok:#ecf7ee; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.58; }
    .page { max-width:1200px; margin:0 auto; padding:36px 22px 64px; }
    .hero { border-bottom:2px solid var(--line); padding-bottom:18px; margin-bottom:24px; }
    h1 { margin:0 0 8px; font-size:30px; letter-spacing:0; }
    h2 { margin:30px 0 12px; font-size:21px; color:var(--accent); }
    p { margin:8px 0; }
    table { border-collapse:collapse; width:100%; background:var(--paper); border:1px solid var(--line); margin:12px 0 20px; font-size:14px; }
    th,td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
    th { background:var(--soft); position:sticky; top:0; }
    .summary { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:14px 0; }
    .card { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .num { font-size:28px; font-weight:800; color:var(--accent); }
    .small { color:var(--muted); font-size:13px; }
    .no { width:56px; font-weight:800; color:var(--accent); text-align:center; }
    .answer { width:64px; text-align:center; font-size:20px; font-weight:800; color:var(--accent2); }
    .tag { display:inline-block; border:1px solid var(--line); background:#fbf6ed; border-radius:999px; padding:2px 8px; margin:2px 3px 2px 0; font-size:12px; color:var(--muted); }
    .options-map { margin-top:8px; display:flex; flex-direction:column; gap:4px; }
    .opt-map { border:1px solid var(--line); border-radius:6px; padding:4px 6px; background:#fff; }
    .opt-map.correct { background:var(--ok); border-color:#b8dfc0; font-weight:700; }
    @media (max-width:860px) { .page{padding:26px 14px 48px;} .summary{grid-template-columns:1fr;} table{font-size:13px;} th,td{padding:7px;} }
    @media print { body{background:#fff;} .page{max-width:none;} th{position:static;} }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>中医 Benchmark 医生盲测答案表</h1>
      <p class="small">对应盲测题本：<code>doctor_sample_20260610_blind.html</code>。本文件仅供内部评分使用。</p>
    </div>

    <div class="summary">
      <div class="card"><div class="num">${answers.length}</div><div class="small">答案总数</div></div>
      <div class="card"><div class="num">${dist.map(row => `${row.key}:${row.count}`).join(' / ')}</div><div class="small">正确选项分布</div></div>
      <div class="card"><div class="num">${benchDist.map(row => `${row.key}:${row.count}`).join(' / ')}</div><div class="small">来源分布</div></div>
    </div>

    <h2>选项分布</h2>
    <table>
      <thead><tr><th>正确选项</th><th>题数</th></tr></thead>
      <tbody>${statRows(dist)}</tbody>
    </table>

    <h2>答案明细</h2>
    <table>
      <thead>
        <tr>
          <th>题号</th>
          <th>答案</th>
          <th>答案文本</th>
          <th>题干</th>
          <th>题目信息与选项映射</th>
        </tr>
      </thead>
      <tbody>${answers.map(answerRow).join('')}</tbody>
    </table>
  </div>
</body>
</html>`;

fs.writeFileSync(outPath, html, 'utf8');
console.log(outPath);
