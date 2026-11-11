// Run with: node tests/test_evaluation_ui.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const script = fs.readFileSync("src/techrag/public/static/app.js", "utf8");
const result = {innerHTML: ""};
const download = {addEventListener() {}};
const context = vm.createContext({
  document: {querySelector: (s) => s === "#evaluation-result" ? result : download},
  dateLabel: (s) => s,
});
const utilities = script.slice(script.indexOf("const $ ="), script.indexOf("const icon ="));
const renderer = script.slice(script.indexOf("const metric ="), script.indexOf("async function openEvaluation"));
vm.runInContext(utilities + renderer, context);
const oldReport = {
  id: "a".repeat(32), created_at: "2026-10-10", notes: [], options: {},
  models: {chat: "mock", embedding: "mock", reranker: "mock"}, corpus_sha256: "test",
  summaries: [{label: "旧方案", completed: 1, cases: 1, recall_at_k: 1, mrr: .5,
    correct_refusal_rate: null, mean_seconds: 2, p95_seconds: 2, degraded_cases: 0,
    calls: {chat_calls: 1, embedding_calls: 1, rerank_calls: 0}}],
  records: [{variant: "vector", case_id: "x", question: "<script>bad()</script>",
    expected_sources: ["a"], retrieved_sources: ["a"], elapsed_seconds: 2,
    error: null, result: {answer: "<img src=x onerror=bad()>", warnings: []}}],
};
context.report = oldReport;
vm.runInContext("renderEvaluation(report)", context);
assert.ok(result.innerHTML.includes("旧报告未记录分阶段耗时"));
assert.ok(result.innerHTML.includes("旧报告未评估回答质量"));
assert.ok(result.innerHTML.includes("&lt;script&gt;"));
assert.ok(!result.innerHTML.includes("<script>bad()"));
assert.ok(!result.innerHTML.includes("<img src=x"));

const report = structuredClone(oldReport);
Object.assign(report.summaries[0], {
  precision_at_k: .5, ndcg_at_k: .6, hit_rate: 1, false_refusal_rate: 0,
  metric_samples: {faithfulness: 1, answer_correctness: 0},
  answer_quality: {faithfulness: .75, answer_correctness: null}, judged_cases: 1,
  judge_failures: 0, judge_calls: 1,
  latency_seconds: {total: {p50: 2, max: 3}, judge: {mean: 1, p95: 1},
    retrieval: {mean: .5, p95: .6}, keyword_search: {mean: .01, p95: .02}},
});
Object.assign(report.records[0], {
  timings_seconds: {retrieval: .5, keyword_search: .01, generation: 1, judge: 1},
  answer_evaluation: {reason: "评审通过", metrics: {faithfulness: .75}},
});
context.report = report;
vm.runInContext("renderEvaluation(report)", context);
assert.ok(result.innerHTML.includes("75.0% <small>n=1</small>"));
assert.ok(result.innerHTML.includes("— <small>n=0</small>"));
assert.ok(result.innerHTML.includes("0.010s / 0.020s"));
assert.ok(result.innerHTML.includes("评审判断、引用证据与参考要点"));
console.log("Evaluation UI: new metrics, legacy reports, timing values and HTML escaping passed.");
