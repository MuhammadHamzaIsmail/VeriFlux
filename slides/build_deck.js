// Builds slides/VeriFlux_presentation.pptx from slides/deck_data.json (run export_data.py first).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const M = Object.fromEntries(D.main.map(r => [r.model, r]));
const g = (m, k, d = 3) => (M[m] && M[m][k] !== undefined ? M[m][k].toFixed(d) : "--");

const THEME = {
  name: "VeriFlux",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF1F5",
    accent1: "0F7173", accent2: "E09F3E", accent3: "9E2A2B", accent4: "5C6B7A",
    accent5: "7FB7BE", accent6: "C9CED6", hlink: "0F7173", folHlink: "5C6B7A",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "VeriFlux";
pres.author = "M. Shah Nawaz, M. Hamza Ismail";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.2, w: 8.8, h: 1.6, fontSize: 34, bold: true, color: C.background1, fontFace: "Cambria", align: "left", valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.0, w: 8.8, h: 1.9, fontSize: 16, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 8.8, h: 0.8, fontSize: 30, bold: true, color: C.background1, fontFace: "Cambria", align: "left", valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.45, w: 8.8, h: 3.6, fontSize: 17, color: C.background2, valign: "top", paraSpaceAfter: 10 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.5, 0.45, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.6, fontSize: 24, bold: true, color: C.text2, fontFace: "Cambria", align: "left", valign: "top", fit: "shrink" }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.22, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent4 },
});

const SECTIONS = ["Introduction", "Background", "Method", "Results", "Closing"];
let sec = null;
function slide(master, section, title, notes) {
  if (section !== sec) { pres.addSection({ title: section }); sec = section; }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function txt(s, text, opt) { s.addText(text, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0.05 }, opt)); }
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function bullets(items, size = 14) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size, paraSpaceAfter: 6 } }));
}
function stat(s, x, y, w, value, label, color) {
  txt(s, value, { x, y, w, h: 0.75, fontSize: 36, bold: true, color, fontFace: "Cambria", valign: "bottom" });
  txt(s, label, { x, y: y + 0.78, w, h: 0.55, fontSize: 12, color: C.accent4 });
}
const chartBase = (title) => ({
  showTitle: false, catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
  valGridLine: { color: "E3E7EC", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFormatCode: "0.000", legendFontFace: "+mn-lt", legendFontSize: 11,
});

// ---------------------------------------------------------------- 1 title
let s = slide("TITLE_DARK", "Introduction", null,
  "Introduce the group and the one-line idea: fake-news detectors on KeepUp look strong on random splits, but most of that comes from memorising events; VeriFlux is a light CPU model that generalises better to unseen events.");
s.addText("VeriFlux: Lightweight, Event-Aware and Explainable Fake News Detection", { placeholder: "title" });
s.addText([
  { text: "Reproducing and improving KeepUp (Array, 2026) on the KeepUp dataset", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "M. Shah Nawaz  ·  M. Hamza Ismail", options: { breakLine: true, bold: true } },
  { text: "Dept. of Software Engineering, FAST-NUCES Chiniot-Faisalabad", options: { breakLine: true } },
  { text: "Social Network Analysis · Fall 2026" },
], { placeholder: "body" });

// ---------------------------------------------------------------- 2 problem
s = slide("CONTENT", "Introduction", "Fake news travels through social signals",
  "Motivation and problem statement. Emphasise that KeepUp fuses five signal types and reports 0.78, but evaluates with one random post split.");
txt(s, bullets([
  "Social posts are judged by the claim, the evidence, who posts and how people react",
  "KeepUp fuses evidence NLI, comment stance, user profile and engagement and reports 0.78 accuracy",
  "But: heavy multi-model pipeline, no cost or explanation, single random post split",
  "Question: does the accuracy survive when a whole news event is unseen?",
]), { x: 0.5, y: 1.25, w: 5.2, h: 3.6 });
[["1,024", "annotated posts (537 fake)"], ["172", "news events with posts"], ["69,507", "stance-labelled comments"]].forEach((v, i) => {
  card(s, 6.1, 1.2 + i * 1.3, 3.4, 1.15, HEX.lt2, "stat" + i);
  stat(s, 6.3, 1.12 + i * 1.3, 3.0, v[0], v[1], C.accent1);
});

// ---------------------------------------------------------------- 3 literature
s = slide("CONTENT", "Background", "Literature: context helps, its cost is unmeasured",
  "16 journal articles 2025-2026. Walk the six themes; the key synthesis is the last card.");
const lit = [
  ["Unified multi-signal", "KeepUp: evidence + stance + user + engagement"],
  ["Lightweight", "EPRVFL, XGBoost studies: fast but text-only"],
  ["User & graph context", "MSDSI-FND, HGCM-Net, SiMiD: strong, heavy"],
  ["Stance & evidence", "HCFND, evolving-stance GNN: extra retrieval cost"],
  ["Unseen events", "Only EvolveDetector tests new events"],
  ["Urdu / multilingual", "Article text only, no social signals"],
];
lit.forEach((v, i) => {
  const x = 0.5 + (i % 3) * 3.05, y = 1.25 + Math.floor(i / 3) * 1.85;
  card(s, x, y, 2.85, 1.65, i === 4 ? "FBEBD2" : HEX.lt2, "lit" + i);
  txt(s, v[0], { x: x + 0.15, y: y + 0.12, w: 2.55, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
  txt(s, v[1], { x: x + 0.15, y: y + 0.55, w: 2.55, h: 1.0, fontSize: 13 });
});

// ---------------------------------------------------------------- 4 gap & objectives
s = slide("CONTENT", "Background", "Research gap and objectives",
  "Gap from the literature review (G1-G5) and the four objectives of this phase.");
card(s, 0.5, 1.2, 4.3, 3.1, HEX.lt2, "gapcard");
txt(s, "Gap", { x: 0.7, y: 1.3, w: 3.9, h: 0.4, fontSize: 18, bold: true, color: C.accent3 });
txt(s, bullets(["No accuracy-and-cost study on a context-rich dataset", "No unseen-event evaluation on KeepUp",
  "Signal contributions entangled", "No validated explanations", "Code-mixed Urdu social signals unexplored"], 13),
  { x: 0.7, y: 1.75, w: 3.9, h: 3.0 });
txt(s, "Objectives", { x: 5.2, y: 1.3, w: 4.3, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
txt(s, bullets(["Re-implement KeepUp and 13 baselines on identical folds", "Quantify event leakage (random vs event split)",
  "Build VeriFlux: light, CPU-only, explainable", "Measure cost, ablation, comment budget, language"], 14),
  { x: 5.2, y: 1.75, w: 4.3, h: 3.0 });

// ---------------------------------------------------------------- 5 base + baselines
s = slide("CONTENT", "Background", "Base paper KeepUp and 13 re-implemented baselines",
  "KeepUp pipeline left to right. Baselines are the families of KeepUp Table 8, all on the same 782-d input.");
const steps = ["BGE-M3 retrieval\n+ re-rank", "NorGLM\nentailment", "XStanceFusion\nstance", "User + title\nfeatures", "CNN + BiGRU\n+ dense fusion"];
steps.forEach((t, i) => {
  const x = 0.5 + i * 1.84;
  card(s, x, 1.3, 1.62, 1.0, i === 4 ? HEX.dk2 : HEX.lt2, "ku" + i);
  txt(s, t, { x: x + 0.05, y: 1.35, w: 1.52, h: 0.9, fontSize: 12, bold: true, align: "center", valign: "middle", color: i === 4 ? C.background1 : C.text2 });
  if (i < 4) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.64, y: 1.7, w: 0.18, h: 0.2, fill: { color: HEX.accent4 }, line: { color: HEX.accent4 } });
});
txt(s, "Baselines (identical folds and features)", { x: 0.5, y: 2.6, w: 9, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
const bl = ["Logistic Regression", "SVM", "Random Forest", "Gradient Boosting", "KNN", "XGBoost", "LSTM", "Bi-RNN", "Bi-GRU", "CNN+LSTM", "XLM-R + LightGBM", "Bi-LSTM + XLM-R", "Voting ensemble"];
bl.forEach((t, i) => {
  const x = 0.5 + (i % 5) * 1.84, y = 3.1 + Math.floor(i / 5) * 0.62;
  card(s, x, y, 1.72, 0.48, i < 6 ? "E3F1F1" : i < 9 ? "FBEBD2" : "ECE3E3", "bl" + i);
  txt(s, t, { x, y, w: 1.72, h: 0.48, fontSize: 12, align: "center", valign: "middle" });
});

// ---------------------------------------------------------------- 6 leakage
s = slide("CONTENT", "Method", "Key finding: random splits reward memorising events",
  "Labels are 83% pure inside an event. A rule that only looks up the event's majority training label matches KeepUp's reported score on random splits and collapses on unseen events. Therefore our primary protocol holds out whole events.");
const pur = D.purity[0] || {};
[[g("Event-majority lookup", "random_f1"), "F1 of a label lookup rule\non a random split", C.accent3],
 [g("Event-majority lookup", "event_f1"), "same rule when events\nare held out", C.accent1],
 [(pur.label_purity_weighted || 0).toFixed(2), "label purity\nwithin an event", C.text2]].forEach((v, i) => {
  card(s, 0.5 + i * 3.05, 1.3, 2.85, 1.65, HEX.lt2, "leak" + i);
  stat(s, 0.7 + i * 3.05, 1.35, 2.5, v[0], v[1], v[2]);
});
txt(s, "Primary protocol: StratifiedGroupKFold by event, 3 × 5 folds. Random post split is reported for comparison with the paper.",
  { x: 0.5, y: 3.3, w: 9, h: 0.9, fontSize: 15, color: C.text2 });

// ---------------------------------------------------------------- 7 framework
s = slide("CONTENT", "Method", "VeriFlux: compare each post with its sibling posts",
  "Two inputs per post: a frozen XLM-R embedding of title + comments and 38 named features. Each is paired with its event-centred copy (minus the mean of posts about the same claim, labels unused). VeriFlux = RBF-SVM on all four. VeriFlux-Lite = LightGBM on named features only, explained with TreeSHAP.");
[["Text embedding", "frozen XLM-R, title + comments"], ["Named features (38)", "evidence, post-claim, stance, user, engagement, title"]].forEach((v, i) => {
  card(s, 0.5 + i * 4.6, 1.2, 4.4, 0.95, "E3F1F1", "inp" + i);
  txt(s, v[0], { x: 0.65 + i * 4.6, y: 1.25, w: 4.1, h: 0.4, fontSize: 15, bold: true, color: C.accent1 });
  txt(s, v[1], { x: 0.65 + i * 4.6, y: 1.65, w: 4.1, h: 0.45, fontSize: 13 });
});
card(s, 0.5, 2.4, 9.0, 0.85, "FBEBD2", "contrast");
txt(s, "Event-context contrast", { x: 0.65, y: 2.45, w: 3.0, h: 0.75, fontSize: 15, bold: true, valign: "middle", color: C.text2 });
txt(s, "each vector minus the mean of all posts about the same claim; labels never used, so it works at inference", { x: 3.6, y: 2.45, w: 5.8, h: 0.75, fontSize: 13, valign: "middle" });
[["VeriFlux", "RBF-SVM on [e, e~, f, f~]: best on unseen events"], ["VeriFlux-Lite", "LightGBM on [f, f~, text prior]: no transformer, TreeSHAP"]].forEach((v, i) => {
  card(s, 0.5 + i * 4.6, 3.5, 4.4, 1.15, i === 0 ? HEX.dk2 : HEX.lt2, "mdl" + i);
  txt(s, v[0], { x: 0.65 + i * 4.6, y: 3.55, w: 4.1, h: 0.4, fontSize: 16, bold: true, color: i === 0 ? C.background1 : C.text2 });
  txt(s, v[1], { x: 0.65 + i * 4.6, y: 3.97, w: 4.1, h: 0.6, fontSize: 13, color: i === 0 ? C.background1 : C.text1 });
});
txt(s, "Output: fake / real probability (+ per-feature reasons for Lite)", { x: 0.5, y: 4.8, w: 9, h: 0.35, fontSize: 12, italic: true, align: "center", color: C.accent4 });

// ---------------------------------------------------------------- 8 setup
s = slide("CONTENT", "Method", "Implementation and experimental setup",
  "Everything CPU-only on a laptop. Same folds for every model; model selection never touches the test fold.");
const rows = [
  ["Hardware", "Intel i7-8665U, 4 cores, 16 GB RAM, no GPU"],
  ["Software", "Python 3.14, PyTorch 2.15, Transformers 5.23, scikit-learn 1.9, LightGBM 4.7"],
  ["Protocols", "Event-held-out 3×5 folds (primary); random post split 3×5 folds"],
  ["Selection", "Inner validation events for early stopping and tuning"],
  ["Stance", "5-fold cross-fitting over events (no own-event leakage)"],
  ["Metrics", "Macro P / R / F1, accuracy, AUC, MCC; Wilcoxon, McNemar, bootstrap CI"],
  ["KeepUp deviations", "Frozen XLM-R (CPU); validation-based early stopping"],
];
s.addTable(rows.map((r, i) => [
  { text: r[0], options: { bold: true, color: HEX.dk2, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } } },
  { text: r[1], options: { color: HEX.dk1, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } } }]),
  { x: 0.5, y: 1.2, w: 9.0, colW: [2.1, 6.9], fontSize: 13, fontFace: "Calibri", rowH: 0.48, border: { type: "none" } });

// ---------------------------------------------------------------- 9 main results chart
s = slide("CONTENT", "Results", "Unseen events: VeriFlux scores highest of 17 models",
  "Event-held-out macro-F1, mean of 15 folds. VeriFlux beats KeepUp clearly (+0.085, p=0.003). Its margin over SVM and Random Forest is small (+0.008) and not statistically significant - say so.");
const order = D.main.filter(r => r.event_f1 !== undefined && r.model !== "Event-majority lookup").sort((a, b) => a.event_f1 - b.event_f1);
s.addChart(pres.charts.BAR, [{ name: "Event-held-out F1", labels: order.map(r => r.model), values: order.map(r => r.event_f1) }],
  Object.assign(chartBase(), { x: 0.4, y: 1.1, w: 6.2, h: 4.2, barDir: "bar", valAxisMinVal: 0.4, valAxisMaxVal: 0.65, showLegend: false,
    chartColors: order.map(r => r.model === "VeriFlux" ? HEX.accent1 : r.model.startsWith("KeepUp") ? HEX.accent2 : HEX.accent6),
    catAxisLabelFontSize: 10, dataLabelPosition: "outEnd" }));
card(s, 6.85, 1.3, 2.65, 3.7, HEX.lt2, "mainstats");
stat(s, 7.0, 1.35, 2.4, g("VeriFlux", "event_f1"), "VeriFlux F1 (event)", C.accent1);
stat(s, 7.0, 2.55, 2.4, g("KeepUp (released code)", "event_f1"), "KeepUp F1 (event)", C.accent2);
stat(s, 7.0, 3.75, 2.4, g("VeriFlux", "event_auc"), "VeriFlux AUC (event)", C.text2);

// ---------------------------------------------------------------- 10 random vs event
s = slide("CONTENT", "Results", "Random splits inflate every model",
  "Grey = random split, teal = event-held-out. VeriFlux-Lite is best on the random split (0.791) but drops most; VeriFlux keeps the best unseen-event score because its contrast features remove event identity.");
const pick = ["Event-majority lookup", "KeepUp (released code)", "KeepUp (paper spec)", "SVM", "Ensemble Learning", "VeriFlux-Lite", "VeriFlux"].filter(m => M[m] && M[m].event_f1 !== undefined);
s.addChart(pres.charts.BAR, [
  { name: "Random split", labels: pick, values: pick.map(m => M[m].random_f1 || 0) },
  { name: "Event-held-out", labels: pick, values: pick.map(m => M[m].event_f1 || 0) }],
  Object.assign(chartBase(), { x: 0.4, y: 1.1, w: 9.2, h: 4.2, barDir: "col", barGrouping: "clustered", valAxisMinVal: 0.3, valAxisMaxVal: 0.85,
    chartColors: [HEX.accent6, HEX.accent1], showLegend: true, legendPos: "t", catAxisLabelFontSize: 10, dataLabelPosition: "outEnd", dataLabelFontSize: 9 }));

// ---------------------------------------------------------------- 11 ablation
s = slide("CONTENT", "Results", "Ablation: the event-centred text view drives the gain",
  "Removing the event contrast or the text embedding lowers F1; the named features add nothing on top of the embedding inside the SVM, which we report openly.");
const abWant = ["VeriFlux w/o text embedding [F, Fc]", "VeriFlux embedding only [E]", "VeriFlux w/o event contrast [E, F]", "VeriFlux w/o named features [E, Ec]"];
const ab = abWant.map(n => D.ablation.find(a => a.model === n)).filter(Boolean);
ab.push({ model: "VeriFlux (full)", f1: M["VeriFlux"] ? M["VeriFlux"].event_f1 : 0 });
s.addChart(pres.charts.BAR, [{ name: "F1", labels: ab.map(a => a.model), values: ab.map(a => a.f1) }],
  Object.assign(chartBase(), { x: 0.4, y: 1.1, w: 6.0, h: 4.2, barDir: "bar", valAxisMinVal: 0.5, valAxisMaxVal: 0.66, showLegend: false,
    chartColors: ab.map(a => a.model === "VeriFlux (full)" ? HEX.accent1 : HEX.accent6), dataLabelPosition: "outEnd" }));
card(s, 6.7, 1.3, 2.8, 2.9, HEX.lt2, "abnote");
txt(s, bullets(["Event contrast: +0.011 F1", "Text embedding: +0.056 F1", "Named features: no extra gain in the SVM", "Lite: claim-level evidence does not transfer to new events"], 13),
  { x: 6.85, y: 1.45, w: 2.5, h: 3.4 });

// ---------------------------------------------------------------- 12 efficiency + budget
s = slide("CONTENT", "Results", "10x cheaper than KeepUp; Lite needs no comments",
  "Left: end-to-end CPU latency per post. Right: VeriFlux-Lite F1 with only the first k comments; comments have no timestamps so this is a proxy for early detection.");
const eff = Object.fromEntries((D.efficiency || []).map(r => [r.component, r]));
const ku = eff["KeepUp (total)"], vf = eff["VeriFlux (total)"];
card(s, 0.5, 1.25, 3.6, 3.75, HEX.lt2, "effcard");
stat(s, 0.7, 1.3, 3.2, ku && vf ? (ku.ms_per_post / vf.ms_per_post).toFixed(1) + "×" : "--", "faster per post than KeepUp\n(CPU, end-to-end)", C.accent1);
stat(s, 0.7, 2.55, 3.2, vf ? Math.round(vf.ms_per_post) + " ms" : "--", "VeriFlux per post", C.text2);
stat(s, 0.7, 3.75, 3.2, ku ? Math.round(ku.ms_per_post) + " ms" : "--", "KeepUp per post", C.accent2);
const bud = D.budget.map(b => ({ k: parseInt(b.model.split("@")[1]), f1: b.f1 })).sort((a, b) => a.k - b.k);
if (M["VeriFlux-Lite"]) bud.push({ k: "all", f1: M["VeriFlux-Lite"].event_f1 });
s.addChart(pres.charts.LINE, [{ name: "F1", labels: bud.map(b => String(b.k)), values: bud.map(b => b.f1) }],
  Object.assign(chartBase(), { x: 4.4, y: 1.15, w: 5.2, h: 4.0, valAxisMinVal: 0.55, valAxisMaxVal: 0.62, showLegend: false,
    chartColors: [HEX.accent1], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7, dataLabelPosition: "t",
    showCatAxisTitle: true, catAxisTitle: "VeriFlux-Lite: comments available per post (first k)", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.accent4 }));

// ---------------------------------------------------------------- 13 explainability
s = slide("CONTENT", "Results", "VeriFlux-Lite explanations are faithful and stable",
  "Global TreeSHAP on held-out events. Deletion test: removing the top-ranked features changes predictions more than removing random ones, so explanations are faithful.");
const sh = (D.shap || []).slice(0, 10).reverse();
s.addChart(pres.charts.BAR, [{ name: "mean |SHAP|", labels: sh.map(r => r[0]), values: sh.map(r => r[1]) }],
  Object.assign(chartBase(), { x: 0.4, y: 1.1, w: 6.2, h: 4.2, barDir: "bar", showLegend: false, chartColors: [HEX.accent1], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00" }));
card(s, 6.85, 1.3, 2.65, 2.4, HEX.lt2, "shapnote");
txt(s, bullets(["Exact TreeSHAP on VeriFlux-Lite", "Top-1 deletion moves p by 0.118 vs 0.016 random", "Rank stability across folds: 0.78"], 13), { x: 7.0, y: 1.45, w: 2.4, h: 3.4 });

// ---------------------------------------------------------------- 14 limitations
s = slide("CONTENT", "Closing", "Limitations and future work",
  "Be honest: unseen-event performance is still modest for every model; the dataset is small.");
card(s, 0.5, 1.2, 4.35, 3.4, HEX.lt2, "limcard");
txt(s, "Limitations", { x: 0.7, y: 1.3, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.accent3 });
txt(s, bullets(["Gain over SVM / RF not significant", "Unseen-event F1 at most 0.63 for every model", "1,024 posts, 172 events: wide confidence intervals", "No comment timestamps; engagement are final totals", "KeepUp encoders frozen (CPU only)", "Heuristic language tags"], 13), { x: 0.7, y: 1.75, w: 4.0, h: 3.1 });
card(s, 5.15, 1.2, 4.35, 3.4, "E3F1F1", "futcard");
txt(s, "Future work", { x: 5.35, y: 1.3, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
txt(s, bullets(["Timestamped re-collection for real early detection", "Fine-tuned multilingual stance on GPU", "Cross-dataset transfer (FakeNewsNet, PHEME)", "Human evaluation of explanations", "Images and video"], 13), { x: 5.35, y: 1.75, w: 4.0, h: 3.1 });

// ---------------------------------------------------------------- 15 conclusion
s = slide("CLOSING_DARK", "Closing", null, "Close with the three contributions and the one number to remember.");
s.addText("Conclusion and contributions", { placeholder: "title" });
s.addText([
  { text: `Event leakage exposed: a label lookup scores ${g("Event-majority lookup", "random_f1")} on random splits, ${g("Event-majority lookup", "event_f1")} on unseen events`, options: { bullet: true, breakLine: true } },
  { text: `VeriFlux is best on unseen events: F1 ${g("VeriFlux", "event_f1")} vs KeepUp ${g("KeepUp (released code)", "event_f1")} (significant) and SVM ${g("SVM", "event_f1")} (not significant)`, options: { bullet: true, breakLine: true } },
  { text: `VeriFlux-Lite is best on the random split: F1 ${g("VeriFlux-Lite", "random_f1")} vs KeepUp ${g("KeepUp (released code)", "random_f1")}`, options: { bullet: true, breakLine: true } },
  { text: "10.6x cheaper than KeepUp per post on CPU; Lite is explainable and works with zero comments", options: { bullet: true } },
], { placeholder: "body" });

(async () => {
  const out = path.join(__dirname, "VeriFlux_presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
