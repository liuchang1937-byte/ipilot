"use strict";
const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const state = {banks: [], sessions: [], mode: "technical", selected: new Set(), topics: new Set(), rounds: 5, difficulty: 2, busy: false, current: null, view: "home", drafts: new Map()};
const library = {banks: [], bankKey: "", active: null, offset: 0, total: 0, limit: 30, drafts: new Map(), requestId: null, bankRequestId: null};
const requestId = () => crypto.randomUUID().replace(/-/g, "");
const phases = {self_introduction: "自我介绍", project_deep_dive: "项目深挖", questioning: "正式问答", candidate_questions: "你的反问", completed: "复盘报告"};
const kinds = {followup: "本题追问", clarification: "题意解释", extend: "引申问题", new_topic: "新的话题", scope_exhausted: "本场题目已完成"};
const dimensions = {knowledge: "知识与经验", reasoning: "分析与推理", depth: "回答深度", communication: "表达沟通", adaptability: "应变与反思"};
const tones = [["#edf2e5", "#749261", "spark"], ["#edf0f8", "#818cb2", "layers"], ["#f8f0e8", "#b08d70", "route"], ["#f0edf5", "#9a88a9", "cpu"]];
let toastTimer;

function connection(online) {
  $("#connection-dot").classList.toggle("offline", !online);
  $("#connection-text").textContent = online ? "服务已连接" : "服务未连接";
}
function toast(message, error = false) {
  const el = $("#toast"); el.textContent = message; el.hidden = false; el.classList.toggle("error", error);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, error ? 7000 : 4000);
}
async function api(path, body, method = "POST") {
  let response;
  try { response = await fetch(path, body === undefined ? {} : {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); }
  catch { connection(false); throw new Error("无法连接面试服务，请检查服务是否仍在运行。"); }
  connection(true);
  const value = await response.json();
  if (!response.ok) { const error = new Error(typeof value.detail === "string" ? value.detail : "提交的内容不符合要求，请检查后重试。"); error.status = response.status; throw error; }
  return value;
}
function busy(value, label = "面试官正在思考你的回答…") {
  state.busy = value;
  $("#thinking").hidden = !value || state.view !== "interview";
  $("#thinking p").textContent = label;
  all(".nav-link, #end-button, #send-button, #open-report, [data-answer], #refresh-banks, #knowledge-send, [data-kb-example], [data-kb-history], .knowledge-settings input").forEach((button) => {button.disabled = value;});
  $("#knowledge-input").disabled = value;
  $("#knowledge-thinking").hidden = !value || state.view !== "knowledge";
  $("#answer-input").disabled = value;
  $("#start-button").disabled = value || state.selected.size === 0;
  all(".bank-card, [data-mode], [data-rounds], [data-difficulty]").forEach((button) => {button.disabled = value;});
  all("#library-view button, #library-view input, #library-view textarea, #library-view select, #bank-dialog button, #bank-dialog input").forEach((el) => {el.disabled = value;});
  $("#editor-thinking").hidden = !value || state.view !== "library" || $("#bank-dialog").open;
  $("#editor-thinking p").textContent = label === "面试官正在思考你的回答…" ? "正在读取题库…" : label;
  $("#bank-create-status").hidden = !value || !$("#bank-dialog").open;
  all("#evaluation-form button, #evaluation-form input, #evaluation-form select").forEach((el) => {el.disabled = value;});
  $("#evaluation-thinking").hidden = !value || state.view !== "evaluation";
  if (!value) { $("#save-question-button").disabled = !library.bankKey; updateLibraryPages(); }
}
function bankName(id, mode = state.mode) { return state.banks.find((b) => b.id === id && b.mode === mode)?.name || id; }
function scopeNames(session) { return session.domains.map((id) => bankName(id, session.mode)).join(" · "); }
function dateLabel(value) {
  if (!value) return "历史练习";
  return new Date(value).toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false});
}
function selectedBanks() { return state.banks.filter((b) => b.mode === state.mode && state.selected.has(b.id)); }
function chip(value) { return `<span class="chip">${escape(value)}</span>`; }

function renderBanks() {
  const banks = state.banks.filter((b) => b.mode === state.mode);
  $("#bank-grid").innerHTML = banks.length ? banks.map((bank, index) => {
    const [bg, tone, symbol] = tones[index % tones.length];
    return `<button class="bank-card ${state.selected.has(bank.id) ? "chosen" : ""}" data-bank="${escape(bank.id)}" aria-pressed="${state.selected.has(bank.id)}" style="--tone-bg:${bg};--tone:${tone}"><span class="bank-icon">${icon(symbol)}</span><span class="bank-selection">${icon("check")}</span><h3>${escape(bank.name)}</h3><div class="bank-description">${escape(bank.topics.slice(0, 3).join(" / "))}</div><div class="bank-meta"><strong>${bank.count} 道题</strong><span>${bank.topics.length} 个知识点</span><span class="difficulty-bars" aria-label="${bank.difficulty_counts.map((c, i) => `${["基础", "标准", "进阶"][i]} ${c}题`).join("，")}">${bank.difficulty_counts.map((count) => `<i class="${count ? "available" : ""}"></i>`).join("")}</span></div></button>`;
  }).join("") : '<div class="empty-state">这个面试类型还没有题库。<br>导入题库后，点击“刷新题库”即可开始练习。</div>';
  renderSetup();
}
function renderSetup() {
  const banks = selectedBanks();
  const availableTopics = [...new Set(banks.flatMap((b) => b.topics))].sort((a, b) => a.localeCompare(b, "zh-CN"));
  state.topics = new Set([...state.topics].filter((p) => availableTopics.includes(p)));
  $("#selected-banks").innerHTML = banks.length ? banks.map((b) => chip(b.name)).join("") : '<span class="selection-placeholder">先在左侧选择题库</span>';
  $("#topic-options").innerHTML = availableTopics.length ? availableTopics.map((topic) => `<label class="topic-option"><input type="checkbox" data-topic="${escape(topic)}" ${state.topics.has(topic) ? "checked" : ""}>${escape(topic)}</label>`).join("") : '<span class="muted">选择题库后，可在这里限定知识点。</span>';
  const count = state.topics.size ? new Set(banks.flatMap((bank) => [...state.topics].flatMap((point) => bank.topic_questions[point] || []))).size : banks.reduce((total, bank) => total + bank.count, 0);
  $("#selected-count").innerHTML = `${count} <small>道</small>`;
  $("#start-button").disabled = state.busy || banks.length === 0;
}
function renderHistory() {
  const recent = state.sessions.slice(0, 3);
  $("#recent-list").innerHTML = recent.length ? recent.map((s) => `<div class="recent-item"><span class="recent-icon">${icon("chat")}</span><div class="recent-info"><strong>${escape(scopeNames(s))}</strong><small>${escape(dateLabel(s.created_at))} · ${s.completed_rounds} / ${s.target_rounds} 题 · ${escape(phases[s.phase] || s.phase)}</small></div><button data-session="${escape(s.id)}" data-destination="${s.phase === "completed" ? "report" : "interview"}">${s.phase === "completed" ? "查看报告" : "继续面试"} ↗</button></div>`).join("") : '<div class="empty-state">还没有面试记录。从第一场练习开始吧。</div>';
  $("#history-list").innerHTML = state.sessions.length ? state.sessions.map((s) => `<article class="history-card"><div class="history-top"><span class="number-label">${escape(dateLabel(s.created_at))}</span><span class="tag ${s.phase === "completed" ? "finished" : ""}">${escape(phases[s.phase] || s.phase)}</span></div><h2>${escape(scopeNames(s))}</h2><p>${s.completed_rounds} / ${s.target_rounds} 道正式问题 · ${s.mode === "technical" ? "技术面试" : "行为面试"}</p><div class="history-bottom"><small>${s.score === null ? "尚无正式题评分" : `综合表现 ${s.score} 分`}</small><button class="outline-button" data-session="${escape(s.id)}" data-destination="${s.phase === "completed" ? "report" : "interview"}">${s.phase === "completed" ? "查看报告" : "继续面试"} ${icon("arrow")}</button></div></article>`).join("") : '<div class="empty-state">还没有面试记录。选择题库，开始第一场练习。</div>';
}
async function refresh() {
  const [banks, sessions] = await Promise.all([api("/api/banks"), api("/api/sessions")]);
  state.banks = banks.banks; state.sessions = sessions.sessions;
  state.selected = new Set([...state.selected].filter((id) => state.banks.some((b) => b.id === id && b.mode === state.mode)));
  renderBanks(); renderHistory();
}
function showView(view) {
  if (state.current && state.view === "interview") state.drafts.set(state.current.session.id, $("#answer-input").value);
  if (state.view === "library") rememberLibraryDraft();
  state.view = view;
  all(".view").forEach((element) => {element.hidden = element.id !== `${view}-view`;});
  const navView = view === "report" ? "history" : view === "interview" ? "home" : view;
  all(".nav-link").forEach((button) => { button.classList.toggle("active", button.dataset.view === navView); });
  $("#page-name").textContent = {home: "模拟面试", interview: "面试对话", history: "面试记录", report: "复盘报告", guide: "使用指南", knowledge: "知识库问答", library: "题库管理", evaluation: "检索评测"}[view];
  window.scrollTo({top: 0, behavior: "instant"});
}
function setHash(view, id) {
  const hash = id ? `#${view}/${id}` : `#${view}`;
  history.replaceState(null, "", hash);
}
function renderConversation(snapshot) {
  state.current = snapshot;
  const s = snapshot.session, finished = s.phase === "completed";
  $("#interview-subtitle").textContent = `${scopeNames(s)} · ${s.target_rounds} 道正式问题 · ${s.difficulty_policy === "adaptive" ? "自适应难度" : "固定难度"}`;
  $("#conversation-status").textContent = kinds[snapshot.step.decision] || phases[s.phase];
  $(".live-tag").innerHTML = finished ? "本场已结束" : '<span class="status-dot"></span>对话中';
  $("#messages").innerHTML = snapshot.messages.map((m) => `<div class="message ${m.role === "user" ? "user" : "assistant"}"><span class="message-avatar">${m.role === "user" ? "你" : icon("spark")}</span><div class="message-body"><div class="message-label">${m.role === "user" ? "你的回答" : escape(kinds[m.kind] || "面试官")}</div><div class="message-bubble">${escape(m.content)}</div></div></div>`).join("");
  const phaseKeys = Object.keys(phases), currentIndex = phaseKeys.indexOf(s.phase);
  $("#stage-list").innerHTML = phaseKeys.map((phase, index) => `<div class="stage ${index === currentIndex ? "active" : index < currentIndex ? "done" : ""}"><span>${index < currentIndex ? icon("check") : String(index + 1).padStart(2, "0")}</span>${phases[phase]}</div>`).join("");
  $("#round-progress").textContent = `${s.completed_rounds} / ${s.target_rounds}`;
  $("#round-progress-bar").style.width = `${Math.min(100, s.completed_rounds / s.target_rounds * 100)}%`;
  $("#session-banks").innerHTML = s.domains.map((id) => chip(bankName(id, s.mode))).join("");
  $("#current-topics").innerHTML = s.current_knowledge_points.length ? s.current_knowledge_points.map(chip).join("") : chip(phases[s.phase]);
  $("#answer-form").hidden = finished; $("#finished-actions").hidden = !finished; $("#end-button").hidden = finished;
  $("#answer-input").value = state.drafts.get(s.id) || ""; updateLength();
  requestAnimationFrame(() => { $("#messages").scrollTop = $("#messages").scrollHeight; });
}
function updateLength() { $("#answer-length").textContent = `${$("#answer-input").value.length} / 12000`; }
async function startInterview() {
  if (state.busy || !state.selected.size) return;
  busy(true);
  try {
    const snapshot = await api("/api/sessions", {mode: state.mode, rounds: state.rounds, difficulty: state.difficulty, difficulty_policy: $("#adaptive").checked ? "adaptive" : "locked", domains: [...state.selected], topics: [...state.topics]});
    showView("interview"); renderConversation(snapshot); setHash("interview", snapshot.session.id);
    await refresh(); $("#answer-input").focus();
  } catch (error) { toast(error.message, true); } finally { busy(false); }
}
async function sendAnswer() {
  const answer = $("#answer-input").value.trim(), current = state.current;
  if (state.busy || !answer || !current) return;
  const id = current.session.id, revision = current.session.turn_index;
  state.drafts.set(id, answer); busy(true);
  try {
    const snapshot = await api(`/api/sessions/${id}/answers`, {answer, expected_turn_index: revision});
    state.drafts.delete(id); renderConversation(snapshot);
  } catch (error) {
    // A lost response may still have committed the answer. Refresh before retrying.
    try {
      const latest = await api(`/api/sessions/${id}`);
      if (latest.session.turn_index !== revision) state.drafts.delete(id);
      renderConversation(latest);
    } catch { /* Keep the unsent draft when the server remains unavailable. */ }
    toast(error.message, true);
  } finally { busy(false); if (state.current?.session.phase !== "completed") $("#answer-input").focus(); }
}
async function openSession(id, destination) {
  if (state.busy) return;
  busy(true);
  try {
    if (destination === "report") await showReport(id);
    else { const snapshot = await api(`/api/sessions/${id}`); showView("interview"); renderConversation(snapshot); setHash("interview", id); }
  } catch (error) { toast(error.message, true); } finally { busy(false); }
}
async function finishInterview() {
  if (!state.current || state.busy) return;
  busy(true, "正在整理你的复盘报告…");
  const id = state.current.session.id;
  try {
    const snapshot = await api(`/api/sessions/${id}/finish`, {expected_turn_index: state.current.session.turn_index});
    state.drafts.delete(id); renderConversation(snapshot); await showReport(id); await refresh();
  } catch (error) { toast(error.message, true); } finally { busy(false); }
}
function scoreRows(values, names) {
  return Object.entries(values).map(([key, score]) => `<div class="score-row"><span>${escape(names(key))}</span><div class="progress-track"><i style="width:${Math.max(0, Math.min(100, score))}%"></i></div><strong>${score}</strong></div>`).join("") || '<p class="muted">本场尚无正式题评分。</p>';
}
async function showReport(id) {
  const report = await api(`/api/sessions/${id}/report`), s = report.session;
  showView("report"); setHash("report", id);
  $("#report-subtitle").textContent = `${scopeNames(s)} · 已完成 ${s.completed_rounds} 道正式问题 · ${escape(dateLabel(s.created_at))}`;
  const reviews = report.reviews.map((review, index) => `<article class="review"><div class="review-title"><span class="number-label">${String(index + 1).padStart(2, "0")}</span><h3>${escape(review.question)}</h3><span class="review-score">${review.score ?? "—"} 分</span></div><p class="answer-text">${escape(review.answer)}</p><p>${escape(review.feedback)}</p>${review.missing_points.length ? `<p>可以补充：${escape(review.missing_points.join("；"))}</p>` : ""}${review.incorrect_points.length ? `<p>需要校正：${escape(review.incorrect_points.join("；"))}</p>` : ""}${pointReviewMarkup(review)}</article>`).join("");
  $("#report-content").innerHTML = `<div class="report-metrics"><div class="metric"><label>综合表现</label><strong>${s.score ?? "—"}<small>${s.score === null ? "暂无评分" : "/ 100"}</small></strong><p>基于已作答的正式问题</p></div><div class="metric"><label>正式问题</label><strong>${s.completed_rounds}<small>/ ${s.target_rounds}</small></strong><p>追问与补充回答合并评估</p></div><div class="metric"><label>本场跳过的知识点</label><strong>${s.blocked_knowledge_points.length}<small>个</small></strong><p>回答中的其他缺口见下方逐题复盘</p></div></div><div class="report-downloads">${report.pdf_ready ? `<a class="primary-button" href="${escape(report.pdf_url)}">${icon("download")} 下载 PDF 报告</a>` : '<span class="muted">PDF 暂未生成，可以先下载 Markdown。</span>'}<a class="outline-button" href="${escape(report.markdown_url)}">${icon("download")} 下载 Markdown</a></div><div class="report-layout"><section class="report-card"><h2>五维能力表现</h2>${scoreRows(report.dimensions, (key) => dimensions[key] || key)}</section><section class="report-card"><h2>知识域表现</h2>${scoreRows(report.domain_scores, (key) => bankName(key, s.mode))}${s.blocked_knowledge_points.length ? `<div class="session-scope"><h3>本场跳过的知识点</h3><div class="chip-list">${s.blocked_knowledge_points.map(chip).join("")}</div></div>` : ""}</section></div><section class="report-card"><h2>逐题复盘</h2>${reviews || '<p class="muted">尚未提交正式题回答，自我介绍和项目交流已保留在完整记录中。</p>'}<details class="report-markdown"><summary>查看完整报告与学习推荐</summary><pre>${escape(report.markdown)}</pre></details></section>`;
}
async function navigate(view) {
  if (state.busy) return;
  showView(view); setHash(view);
  if (view === "history" || view === "home") try { await refresh(); } catch (error) { toast(error.message, true); }
  if (view === "knowledge") try { await loadKnowledge(); } catch (error) { toast(error.message, true); }
  if (view === "library") try { await loadLibrary(); } catch (error) { toast(error.message, true); }
  if (view === "evaluation") try { await loadEvaluations(); } catch (error) { toast(error.message, true); }
}
document.addEventListener("click", async (event) => {
  const button = event.target.closest("button, a"); if (!button || state.busy) return;
  if (button.dataset.view) await navigate(button.dataset.view);
  else if (button.dataset.bank) { const id = button.dataset.bank; state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id); renderBanks(); }
  else if (button.dataset.mode) { state.mode = button.dataset.mode; state.selected.clear(); state.topics.clear(); all("[data-mode]").forEach((b) => {b.classList.toggle("selected", b.dataset.mode === state.mode);}); renderBanks(); }
  else if (button.dataset.rounds) { state.rounds = Number(button.dataset.rounds); all("[data-rounds]").forEach((b) => {b.classList.toggle("selected", Number(b.dataset.rounds) === state.rounds);}); }
  else if (button.dataset.difficulty) { state.difficulty = Number(button.dataset.difficulty); all("[data-difficulty]").forEach((b) => {b.classList.toggle("selected", Number(b.dataset.difficulty) === state.difficulty);}); }
  else if (button.dataset.session) await openSession(button.dataset.session, button.dataset.destination);
  else if (button.dataset.answer) { $("#answer-input").value = button.dataset.answer; updateLength(); $("#answer-input").focus(); }
  else if (button.dataset.kbExample) { $("#knowledge-input").value = button.dataset.kbExample; $("#knowledge-input").focus(); }
  else if (button.dataset.kbHistory) await openKnowledge(button.dataset.kbHistory);
  else if (button.dataset.editQuestion) await editLibraryQuestion(button.dataset.editQuestion);
});
document.addEventListener("change", (event) => { if (event.target.dataset.topic) { const topic = event.target.dataset.topic; event.target.checked ? state.topics.add(topic) : state.topics.delete(topic); renderSetup(); } });
$("#start-button").addEventListener("click", startInterview);
$("#refresh-banks").addEventListener("click", async () => { try { await refresh(); toast("题库已更新。"); } catch (error) { toast(error.message, true); } });
$("#answer-form").addEventListener("submit", (event) => {event.preventDefault(); sendAnswer();});
$("#answer-input").addEventListener("input", updateLength);
$("#answer-input").addEventListener("keydown", (event) => { if (!event.isComposing && (event.ctrlKey || event.metaKey) && event.key === "Enter") {event.preventDefault(); sendAnswer();} });
$("#end-button").addEventListener("click", () => { $("#finish-dialog").returnValue = "cancel"; $("#finish-dialog").showModal(); });
$("#finish-dialog").addEventListener("close", () => { if ($("#finish-dialog").returnValue === "finish") finishInterview(); });
$("#open-report").addEventListener("click", () => openSession(state.current.session.id, "report"));
$(".brand").addEventListener("click", (event) => {event.preventDefault(); navigate("home");});
window.addEventListener("hashchange", route);
async function route() {
  const [view = "home", id] = location.hash.slice(1).split("/");
  if (id && ["interview", "report"].includes(view)) await openSession(id, view);
  else if (view === "knowledge") { showView(view); await loadKnowledge(); if (id) await openKnowledge(id); }
  else if (view === "library") { showView(view); await loadLibrary(); }
  else if (view === "evaluation") { showView(view); await loadEvaluations(); if (id) await openEvaluation(id); }
  else if (["home", "history", "guide"].includes(view)) showView(view);
}
async function loadKnowledge() {
  const [info, journal] = await Promise.all([api("/api/knowledge"), api("/api/knowledge/history")]);
  $("#knowledge-count").textContent = `${info.documents} 份资料 · ${info.parent_chunks} 个完整资料块`;
  const previous = new Set(all("[data-kb-domain]:checked").map((el) => el.value));
  $("#knowledge-domains").innerHTML = info.domains.map((d) => `<label><input type="checkbox" data-kb-domain value="${escape(d.id)}" ${previous.has(d.id) ? "checked" : ""}>${escape(d.name)}</label>`).join("") || '<span class="muted">尚未导入带方向的资料。</span>';
  $("#kb-reranker-status").textContent = info.reranker_ready ? "本地多语言模型已就绪" : "本地模型未就绪，当前可使用混合检索";
  $("#knowledge-history").innerHTML = journal.items.length ? journal.items.slice(0, 10).map((item) => `<button data-kb-history="${escape(item.id)}"><span>${escape(item.question)}</span><small>${escape(dateLabel(item.created_at))}</small></button>`).join("") : '<p class="muted">你的问题会保存在这里。</p>';
}
function knowledgeMarkup(text, id) {
  return String(text).split(/```[^\n]*\n([\s\S]*?)```/g).map((part, index) => {
    if (index % 2) return `<pre class="knowledge-code"><code>${escape(part)}</code></pre>`;
    return escape(part).replace(/^#{1,4}\s+(.+)$/gm, '<strong class="knowledge-subheading">$1</strong>').replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>').replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\[(\d+)\]/g, (_, n) => `<a class="knowledge-citation" href="#kb-source-${escape(id)}-${n}" data-citation="${n}">[${n}]</a>`);
  }).join("");
}
function renderKnowledge(result) {
  const trace = result.sections.filter((s) => s.kind === "knowledge");
  const precise = trace.filter((s) => s.trace.reranked).length;
  const questions = result.sections.map((section, i) => `<section class="knowledge-section"><div class="knowledge-section-title"><span class="number-label">${String(i + 1).padStart(2, "0")}</span><h3>${escape(section.question)}</h3>${section.insufficient ? '<span class="tag">资料不足</span>' : ""}</div><div class="knowledge-answer-text">${knowledgeMarkup(section.answer, result.id)}</div>${section.source_ids.length ? `<div class="knowledge-reference-row"><span>本题依据</span>${section.source_ids.map((n) => `<a href="#kb-source-${escape(result.id)}-${n}" data-citation="${n}" class="chip">[${n}] ${escape(result.sources.find((s) => s.id === n)?.title || "参考资料")}</a>`).join("")}</div>` : ""}</section>`).join("");
  const sources = result.sources.map((source) => `<details class="knowledge-source" id="kb-source-${escape(result.id)}-${source.id}"><summary><span>[${source.id}]</span><strong>${escape(source.title)}</strong>${icon("arrow")}</summary><div class="knowledge-source-meta">${escape(source.domain ? bankName(source.domain) : "知识库资料")} · ${escape(source.source)}${source.rerank_score !== null ? ` · 精排得分 ${source.rerank_score.toFixed(3)}` : ""}</div><pre>${escape(source.content)}</pre></details>`).join("");
  const steps = trace.map((section, i) => `<article class="knowledge-trace-item"><h4>${i + 1}. ${escape(section.question)}</h4><p>BM25 + 向量检索 → RRF 融合 → ${section.trace.reranked ? "Cross Encoder 精排" : "混合检索排序"} → ${section.trace.selected_count} 个完整资料块</p>${section.trace.queries.map((query) => `<div><span class="tag">${{original: "原问题", mqe: "MQE", hyde: "HyDE · 非证据"}[query.kind] || escape(query.kind)}</span><span>${escape(query.text)}</span></div>`).join("")}</article>`).join("");
  $("#knowledge-result").innerHTML = `<article class="knowledge-answer-card"><div class="knowledge-answer-top"><span class="number-label">ANSWER / ${escape(dateLabel(result.created_at))}</span><span class="tag">${result.sections.length} 个问题 · ${precise} 次精排</span></div><h2>${escape(result.question)}</h2>${result.warnings.length ? `<div class="knowledge-warning">${result.warnings.map(escape).join("<br>")}</div>` : ""}${questions}${sources ? `<div class="knowledge-source-list"><h3>参考资料 · ${result.sources.length}</h3>${sources}</div>` : ""}<details class="knowledge-trace"><summary>查看问题拆解与检索过程</summary>${steps}</details></article>`;
  all("[data-citation]").forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault(); const details = document.getElementById(`kb-source-${result.id}-${link.dataset.citation}`);
    if (details) { details.open = true; details.scrollIntoView({behavior: "smooth", block: "center"}); }
  }));
}
async function askKnowledge() {
  const question = $("#knowledge-input").value.trim();
  if (state.busy || !question) return;
  busy(true);
  try {
    const result = await api("/api/knowledge/ask", {question, domains: all("[data-kb-domain]:checked").map((el) => el.value), decompose: $("#kb-decompose").checked, mqe: $("#kb-mqe").checked, hyde: $("#kb-hyde").checked, rerank: $("#kb-rerank").checked});
    renderKnowledge(result); setHash("knowledge", result.id); await loadKnowledge();
  } catch (error) { toast(error.message, true); } finally { busy(false); }
}
async function openKnowledge(id) {
  if (state.busy) return;
  busy(true);
  try { const result = await api(`/api/knowledge/answers/${encodeURIComponent(id)}`); renderKnowledge(result); $("#knowledge-input").value = result.question; setHash("knowledge", id); }
  catch (error) { toast(error.message, true); } finally { busy(false); }
}
$("#knowledge-form").addEventListener("submit", (event) => {event.preventDefault(); askKnowledge();});
$("#knowledge-input").addEventListener("keydown", (event) => {if (!event.isComposing && (event.ctrlKey || event.metaKey) && event.key === "Enter") {event.preventDefault(); askKnowledge();}});
function rememberLibraryDraft() {
  if (!library.bankKey) return;
  const key = library.active?.id || `new:${library.bankKey}`;
  library.drafts.set(key, {question: $("#editor-question").value, answer: $("#editor-answer").value, requestId: library.requestId});
}
function renderLibraryEditor() {
  const item = library.active, draft = library.drafts.get(item?.id || `new:${library.bankKey}`);
  $("#editor-question").value = draft?.question ?? item?.question ?? "";
  $("#editor-answer").value = draft?.answer ?? item?.answer ?? "";
  library.requestId = draft?.requestId || requestId();
  $("#editor-status").textContent = item ? "编辑现有题目" : "新增题目";
  $("#editor-heading").textContent = item ? "把这道题，整理得更清楚。" : "写下一道值得练习的题目。";
  $("#save-question-button").disabled = state.busy || !library.bankKey;
  const generated = $("#editor-generated"); generated.hidden = !item;
  if (item) generated.innerHTML = `<details open><summary>${item.online ? "AI 已补全的辅助信息" : "已有题目辅助信息"}</summary><div class="editor-meta"><span class="chip">${["", "基础", "标准", "进阶"][item.difficulty]}</span>${item.tags.map(chip).join("")}</div><h3>知识点</h3><div class="chip-list">${item.knowledge_points.length ? item.knowledge_points.map(chip).join("") : '<span class="muted">保存修改后，AI 会补全知识点。</span>'}</div><h3>前置知识</h3><p>${escape(item.prerequisites.join("、") || "暂无")}</p><h3>评分要点</h3><p>${escape(item.scoring_points)}</p><small>辅助信息会根据你的题目和答案生成；题目、答案原文由你决定。</small></details>`;
}
async function loadLibrary() {
  const data = await api("/api/library/banks"); library.banks = data.banks;
  if (!library.banks.some((b) => b.key === library.bankKey)) { library.bankKey = library.banks[0]?.key || ""; library.active = null; }
  $("#library-bank").innerHTML = library.banks.length ? library.banks.map((bank) => `<option value="${escape(bank.key)}" ${library.bankKey === bank.key ? "selected" : ""}>${escape(bank.name)} · ${bank.count} 题</option>`).join("") : '<option value="">先创建一个题库</option>';
  await loadLibraryQuestions(); renderLibraryEditor();
}
async function loadLibraryQuestions() {
  if (!library.bankKey) { $("#library-questions").innerHTML = '<div class="empty-state">创建一个新题库，添加你的第一道题。</div>'; library.total = 0; updateLibraryPages(); return; }
  const search = $("#library-search").value.trim();
  const data = await api(`/api/library/banks/${encodeURIComponent(library.bankKey)}/questions?search=${encodeURIComponent(search)}&offset=${library.offset}&limit=${library.limit}`);
  library.total = data.total;
  $("#library-description").textContent = data.bank.description;
  $("#library-count").textContent = `${data.total} 题`;
  $("#library-questions").innerHTML = data.questions.length ? data.questions.map((q) => `<button class="library-question ${library.active?.id === q.id ? "selected" : ""}" data-edit-question="${escape(q.id)}"><strong>${escape(q.question)}</strong><span>${["", "基础", "标准", "进阶"][q.difficulty]} · ${escape(q.knowledge_points.slice(0, 2).join(" / ") || "已有题目")}</span></button>`).join("") : '<div class="empty-state">没有匹配的题目。<br>可以点击“新增”添加一道题。</div>';
  $("#library-export").hidden = data.bank.count === 0;
  $("#library-export").href = `/api/library/banks/${encodeURIComponent(library.bankKey)}/export`;
  updateLibraryPages();
}
function updateLibraryPages() {
  $("#library-prev").disabled = state.busy || library.offset === 0;
  $("#library-next").disabled = state.busy || library.offset + library.limit >= library.total;
  $("#library-page").textContent = library.total ? `${Math.floor(library.offset / library.limit) + 1} / ${Math.ceil(library.total / library.limit)}` : "0 / 0";
}
async function editLibraryQuestion(id) {
  if (state.busy) return; rememberLibraryDraft(); busy(true);
  try { library.active = await api(`/api/library/questions/${encodeURIComponent(id)}`); renderLibraryEditor(); all("[data-edit-question]").forEach((el) => el.classList.toggle("selected", el.dataset.editQuestion === id)); }
  catch (error) { toast(error.message, true); } finally { busy(false); }
}
function newLibraryQuestion() { if (state.busy) return; rememberLibraryDraft(); library.active = null; renderLibraryEditor(); all("[data-edit-question]").forEach((el) => el.classList.remove("selected")); $("#editor-question").focus(); }
async function saveLibraryQuestion() {
  if (state.busy || !library.bankKey) return;
  const question = $("#editor-question").value.trim(), answer = $("#editor-answer").value.trim();
  if (!question || !answer) { toast("请填写题目和参考答案。", true); return; }
  const current = library.active, token = library.requestId || requestId();
  const id = current?.id || `Q${BigInt(`0x${token}`).toString()}`;
  const draftKey = current?.id || `new:${library.bankKey}`;
  library.requestId = token; rememberLibraryDraft(); busy(true, "正在生成辅助信息并更新索引…");
  let result;
  try {
    const body = {question, answer, request_id: token};
    if (current) body.expected_revision = current.revision;
    result = await api(current ? `/api/library/questions/${id}` : `/api/library/banks/${encodeURIComponent(library.bankKey)}/questions`, body, current ? "PUT" : "POST");
  } catch (error) {
    // Recover a completed save when only the response was lost. Keep the draft
    // and its revision otherwise, so a retry cannot overwrite newer content.
    try { const recovered = await api(`/api/library/questions/${id}`); if (recovered.question === question && recovered.answer === answer) result = recovered; } catch { /* Draft retained. */ }
    if (!result) toast(error.message, true);
  }
  if (result) {
    library.drafts.delete(draftKey); library.drafts.delete(id); library.active = result; library.offset = 0;
    renderLibraryEditor();
    try { await loadLibrary(); await refresh(); } catch { toast("题目已保存；列表暂未刷新，可稍后重新打开题库。", true); }
    toast("题目已保存，辅助信息和检索索引已更新。");
  }
  busy(false);
}
async function createLibraryBank(event) {
  event.preventDefault(); if (state.busy) return;
  const name = $("#bank-name-input").value.trim(); if (!name) return;
  library.bankRequestId ||= requestId(); rememberLibraryDraft(); busy(true);
  try {
    const bank = await api("/api/library/banks", {name, request_id: library.bankRequestId});
    library.bankKey = bank.key; library.active = null; library.offset = 0; $("#library-search").value = "";
    $("#bank-dialog").close(); await loadLibrary(); toast("题库已创建，现在可以添加第一道题。");
  } catch (error) { toast(error.message, true); } finally { busy(false); }
}
$("#create-bank-button").addEventListener("click", () => {if (!state.busy) { library.bankRequestId = requestId(); $("#bank-name-input").value = ""; $("#bank-dialog").showModal(); }});
$("#bank-cancel-button").addEventListener("click", () => {if (!state.busy) $("#bank-dialog").close();});
$("#bank-dialog").addEventListener("cancel", (event) => {if (state.busy) event.preventDefault();});
$("#bank-name-input").addEventListener("input", () => {library.bankRequestId = requestId();});
$("#bank-create-form").addEventListener("submit", createLibraryBank);
$("#question-editor-form").addEventListener("submit", (event) => {event.preventDefault(); saveLibraryQuestion();});
all("#editor-question, #editor-answer").forEach((el) => el.addEventListener("input", () => {library.requestId = requestId();}));
$("#new-question-button").addEventListener("click", newLibraryQuestion);
$("#library-bank").addEventListener("change", async () => {if (state.busy) return; rememberLibraryDraft(); library.bankKey = $("#library-bank").value; library.active = null; library.offset = 0; $("#library-search").value = ""; busy(true); try {await loadLibraryQuestions(); renderLibraryEditor();} catch (error) {toast(error.message, true);} finally {busy(false);} });
$("#library-search-form").addEventListener("submit", async (event) => {event.preventDefault(); if (state.busy) return; library.offset = 0; busy(true); try {await loadLibraryQuestions();} catch (error) {toast(error.message, true);} finally {busy(false);} });
for (const [selector, amount] of [["#library-prev", -1], ["#library-next", 1]]) $(selector).addEventListener("click", async () => {if (state.busy) return; library.offset += amount * library.limit; busy(true); try {await loadLibraryQuestions();} catch (error) {toast(error.message, true);} finally {busy(false);} });
function pointReviewMarkup(review) {
  if (!review.point_reviews?.length) return '<p class="muted">这条历史评价未记录逐项依据。</p>';
  return `<details class="point-reviews" open><summary>逐项评分依据${review.scoring_method === "fallback" ? " · 降级评分" : ""}</summary>${review.point_reviews.map((p) => `<div class="point-review"><div><span class="tag point-${escape(p.status)}">${escape(p.label)}</span><strong>${escape(p.point)}</strong></div><p>${escape(p.reason)}</p>${p.evidence.map((e) => `<blockquote><small>第 ${escape(e.answer_index)} 次回答原文</small>${escape(e.quote)}</blockquote>`).join("")}</div>`).join("")}</details>`;
}
const metric = (value) => value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
async function loadEvaluations() {
  const data = await api("/api/evaluations");
  $("#evaluation-dataset").textContent = `${data.dataset.name} · ${data.case_count} 个案例`;
  const selected = new Set(all("[data-eval-variant]:checked").map((el) => el.value));
  const initialized = $("#evaluation-variants").children.length > 0;
  $("#evaluation-variants").innerHTML = data.variants.map((v) => `<label class="chip"><input type="checkbox" data-eval-variant value="${escape(v.id)}" ${(initialized ? selected.has(v.id) : v.default !== false) ? "checked" : ""}> ${escape(v.label)}</label>`).join("");
  $("#evaluation-history").innerHTML = data.reports.length ? data.reports.map((r) => `<button class="outline-button" data-eval-history="${escape(r.id)}">${escape(dateLabel(r.created_at))} · ${r.dataset.selected_cases} 个案例 · ${r.summaries.length} 个方案</button>`).join("") : '<p class="muted">运行一次对比后，结果会保存在这里。</p>';
}
const evaluationStages = [["decomposition", "共有拆解"], ["retrieval", "检索总计（含以下子阶段）"], ["mqe", "MQE"], ["hyde", "HyDE"], ["embedding", "Embedding"], ["vector_search", "向量检索"], ["keyword_search", "BM25"], ["rerank", "精排"], ["generation", "答案生成与校验"], ["judge", "答案评审（另计）"]];
const answerMetricLabels = [["faithfulness", "忠实度"], ["answer_correctness", "正确性 F1"], ["answer_relevancy", "回答相关性"], ["citation_precision", "引用准确性"], ["citation_coverage", "引用覆盖率"]];
const seconds = (v) => typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(3)}s` : "—";
const evaluationTable = (headers, rows) => `<div class="evaluation-table"><table><thead><tr>${headers.map((h) => `<th>${escape(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
function evaluationScore(s, key, value, decimal = false) {
  const n = s.metric_samples?.[key];
  const text = decimal ? (typeof value === "number" ? value.toFixed(3) : "—") : metric(value);
  return `${text}${n === undefined ? "" : ` <small>n=${n}</small>`}`;
}
function evaluationCase(r) {
  const timings = r.timings_seconds;
  const latency = timings ? `<p>${evaluationStages.map(([key, label]) => `${escape(label)}：${seconds(timings[key])}`).join(" · ")}</p>` : '<p class="muted">旧报告未记录分阶段耗时。</p>';
  const scores = `<p>Recall：${metric(r.recall_at_k)} · Precision：${metric(r.precision_at_k)} · Hit：${r.hit === undefined || r.hit === null ? "—" : r.hit ? "命中" : "未命中"} · nDCG：${metric(r.ndcg_at_k)} · RR：${metric(r.reciprocal_rank)}</p>`;
  const quality = r.answer_evaluation;
  const review = quality ? `<p>答案评审：${escape(quality.reason)}</p><p>${answerMetricLabels.map(([key, label]) => `${label}：${metric(quality.metrics?.[key])}`).join(" · ")}</p><details><summary>评审判断、引用证据与参考要点</summary><pre>${escape(JSON.stringify(quality, null, 2))}</pre></details>` : '<p class="muted">旧报告未评估回答质量。</p>';
  return `<details class="evaluation-case"><summary><span class="tag">${escape(r.variant)}</span> ${escape(r.case_id)} · ${escape(r.question)} <small>问答 ${seconds(r.elapsed_seconds)}</small></summary>${latency}${scores}<p>标注来源：${escape(r.expected_sources.join("、") || "无，预期拒答")}</p><p>检索来源：${escape((r.retrieved_sources || []).join("、") || "无")}</p>${r.error ? `<p>${escape(r.error)}</p>` : `<div class="answer-text">${escape(r.result.answer)}</div>${r.result.warnings.map((w) => `<p>${escape(w)}</p>`).join("")}<details><summary>来源全文与检索轨迹</summary><pre>${escape(JSON.stringify(r.result, null, 2))}</pre></details>`}${review}</details>`;
}
function renderEvaluation(report) {
  const retrieval = evaluationTable(["方案", "完成", "来源 Recall@3", "标注 Precision@3", "Hit Rate@3", "MRR", "nDCG@3", "正确拒答", "误拒答"], report.summaries.map((s) => [escape(s.label), `${s.completed}/${s.cases}`, evaluationScore(s, "recall_at_k", s.recall_at_k), evaluationScore(s, "precision_at_k", s.precision_at_k), evaluationScore(s, "hit", s.hit_rate), evaluationScore(s, "reciprocal_rank", s.mrr, true), evaluationScore(s, "ndcg_at_k", s.ndcg_at_k), evaluationScore(s, "correct_refusal", s.correct_refusal_rate), evaluationScore(s, "false_refusal", s.false_refusal_rate)]));
  const quality = evaluationTable(["方案", ...answerMetricLabels.map(([, label]) => label), "评审完成 / 失败"], report.summaries.map((s) => [escape(s.label), ...answerMetricLabels.map(([key]) => evaluationScore(s, key, s.answer_quality?.[key])), `${s.judged_cases ?? "—"} / ${s.judge_failures ?? "—"}`]));
  const latency = evaluationTable(["方案", "问答平均", "P50", "P95", "最大", "评审平均（另计）", "LLM / Embedding / 精排调用", "评审调用"], report.summaries.map((s) => [escape(s.label), seconds(s.mean_seconds), seconds(s.latency_seconds?.total?.p50), seconds(s.p95_seconds), seconds(s.latency_seconds?.total?.max), seconds(s.latency_seconds?.judge?.mean), `${s.calls.chat_calls} / ${s.calls.embedding_calls} / ${s.calls.rerank_calls}`, `${s.judge_calls ?? "—"}`]));
  const stages = evaluationTable(["阶段（均值 / P95）", ...report.summaries.map((s) => s.label)], evaluationStages.map(([key, label]) => [escape(label), ...report.summaries.map((s) => {const t = s.latency_seconds?.[key]; return `${seconds(t?.mean)} / ${seconds(t?.p95)}`;})]));
  $("#evaluation-result").innerHTML = `<section class="report-card"><h2>本次对比 · ${escape(dateLabel(report.created_at))}</h2><h3>检索质量</h3>${retrieval}<h3>回答质量</h3><p class="muted">n 为有效样本数；— 表示未评估。答案指标来自独立评审调用，默认复用生成模型，需人工抽查。</p>${quality}<h3>具体时延与调用量</h3><p class="muted">问答耗时包含共有拆解，不含答案评审。检索总计包含 MQE、HyDE、Embedding、向量/BM25 等子阶段，不能重复相加。</p>${latency}<details><summary>分阶段时延</summary>${stages}</details><details class="evaluation-notes"><summary>指标说明与本次配置</summary>${report.notes.map((n) => `<p>${escape(n)}</p>`).join("")}<p>题库指纹：${escape(report.corpus_sha256)}<br>模型：${escape(report.models.chat)} · ${escape(report.models.embedding)} · ${escape(report.models.reranker)}</p><pre>${escape(JSON.stringify(report.options, null, 2))}</pre></details><button class="text-button" id="evaluation-download">下载完整 JSON ↗</button></section><section class="report-card"><h2>逐题结果与原始证据</h2>${report.records.map(evaluationCase).join("")}</section>`;
  $("#evaluation-download").addEventListener("click", () => {
    const link = document.createElement("a"); link.href = `/api/evaluations/${report.id}/download`; link.download = `evaluation-${report.id}.json`; document.body.append(link); link.click(); link.remove();
  });
}
async function openEvaluation(id) { const report = await api(`/api/evaluations/${id}`); renderEvaluation(report); setHash("evaluation", id); }
$("#evaluation-history").addEventListener("click", async (event) => {const el = event.target.closest("[data-eval-history]"); if (!el || state.busy) return; try {await openEvaluation(el.dataset.evalHistory);} catch (error) {toast(error.message, true);} });
$("#evaluation-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (state.busy) return;
  const variants = all("[data-eval-variant]:checked").map((el) => el.value);
  if (!variants.length) {toast("请至少选择一个检索方案。", true); return;}
  busy(true);
  try {const report = await api("/api/evaluations/run", {limit:Number($("#evaluation-limit").value), variants, answer_judge:$("#evaluation-answer-judge").checked}); renderEvaluation(report); setHash("evaluation", report.id); await loadEvaluations(); toast("对比完成，已保存逐题结果。");}
  catch (error) {toast(error.message, true);} finally {busy(false);}
});
$("#today").textContent = new Date().toLocaleDateString("zh-CN", {month: "long", day: "numeric", weekday: "long"});
(async () => { try { await refresh(); await route(); } catch (error) { $("#bank-grid").innerHTML = '<div class="empty-state">暂时无法读取题库。请检查服务后点击“刷新题库”。</div>'; toast(error.message, true); } })();
