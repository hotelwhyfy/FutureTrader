/* Stokker UI.
   Charts are hand-drawn SVG rather than a library: the whole app is four chart
   shapes, and this keeps the page dependency-free and fully offline. */

const SIGNAL_LABELS = {};
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVG_NS = "http://www.w3.org/2000/svg";

/* ------------------------------------------------------------ formatting */
const isNum = v => v !== null && v !== undefined && Number.isFinite(v);
const fmt = {
  pct:  (v, d = 1) => isNum(v) ? (v * 100).toFixed(d) + "%" : "–",
  sig:  (v, d = 1) => isNum(v) ? (v > 0 ? "+" : "") + (v * 100).toFixed(d) + "%" : "–",
  num:  (v, d = 2) => isNum(v) ? v.toFixed(d) : "–",
  int:  v => isNum(v) ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "–",
  x:     v => isNum(v) ? v.toFixed(2) + "×" : "–",
  // Futures prices span 0.0067 (JPY) to 29,374 (NQ); fixed precision breaks at
  // both ends, so scale decimals to magnitude.
  price: v => {
    if (!isNum(v)) return "–";
    const a = Math.abs(v);
    const d = a >= 1000 ? 0 : a >= 100 ? 2 : a >= 10 ? 2 : a >= 1 ? 4 : 6;
    return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  },
};
const cls = v => !isNum(v) ? "dim" : v > 0 ? "pos" : v < 0 ? "neg" : "dim";

/* ------------------------------------------------------------------- api */
async function api(path, params = {}) {
  const q = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
  );
  const res = await fetch(`/api/${path}?${q}`);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const opts = () => ({
  signal: $("#signal").value,
  start: $("#start").value,
  stressed: $("#stressed").checked,
});

const busy = (node, msg) => { node.innerHTML = `<div class="loading"><span class="spinner"></span>${msg}</div>`; };
const fail = (node, e) => { node.innerHTML = `<div class="error">${e.message ?? e}</div>`; };

/* ---------------------------------------------------------- svg helpers */
function svgEl(tag, attrs = {}) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function niceTicks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max) || min === max) return [min ?? 0];
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) ?? mag * 10;
  const out = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 0.001; t += step) out.push(t);
  return out;
}

const CHARTS = new Map();   // container -> spec, so resize can re-render

/* A multi-series line chart with optional log scale, area fill and crosshair. */
function lineChart(container, spec) {
  CHARTS.set(container, spec);
  const {
    dates, series, height = 300, logScale = false,
    yFormat = v => fmt.num(v, 2), zeroLine = false, area = false,
  } = spec;

  const W = Math.max(container.clientWidth || 900, 320);
  const H = height;
  const PAD = { t: 12, r: 14, b: 26, l: 56 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;

  container.innerHTML = "";
  const svg = svgEl("svg", { class: "chart", width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });

  const all = series.flatMap(s => s.values).filter(isNum);
  if (!all.length) { container.innerHTML = `<div class="empty">No data.</div>`; return; }

  let lo = Math.min(...all), hi = Math.max(...all);
  if (logScale) { lo = Math.max(lo, 1e-6); }
  if (zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.06 || 1;
  lo -= pad; hi += pad;
  if (logScale) lo = Math.max(lo, Math.min(...all) * 0.9, 1e-6);

  const proj = v => logScale ? Math.log(Math.max(v, 1e-9)) : v;
  const y0 = proj(lo), y1 = proj(hi);
  const sy = v => PAD.t + ih - ((proj(v) - y0) / (y1 - y0 || 1)) * ih;
  const sx = i => PAD.l + (dates.length < 2 ? iw / 2 : (i / (dates.length - 1)) * iw);

  // grid + y labels
  const ticks = logScale
    ? niceTicks(lo, hi, 4)
    : niceTicks(lo, hi, 5);
  for (const t of ticks) {
    const y = sy(t);
    if (y < PAD.t - 1 || y > PAD.t + ih + 1) continue;
    svg.append(svgEl("line", { class: "grid-line", x1: PAD.l, x2: W - PAD.r, y1: y, y2: y }));
    const lab = svgEl("text", { class: "axis-label", x: PAD.l - 7, y: y + 3.5, "text-anchor": "end" });
    lab.textContent = yFormat(t);
    svg.append(lab);
  }
  if (zeroLine && lo < 0 && hi > 0) {
    svg.append(svgEl("line", { class: "zero", x1: PAD.l, x2: W - PAD.r, y1: sy(0), y2: sy(0) }));
  }

  // x labels
  const nx = Math.min(6, dates.length);
  for (let k = 0; k < nx; k++) {
    const i = Math.round((k / Math.max(nx - 1, 1)) * (dates.length - 1));
    const lab = svgEl("text", {
      class: "axis-label", x: sx(i), y: H - 8,
      "text-anchor": k === 0 ? "start" : k === nx - 1 ? "end" : "middle",
    });
    lab.textContent = (dates[i] || "").slice(0, 7);
    svg.append(lab);
  }

  // series
  for (const s of series) {
    const pts = [];
    s.values.forEach((v, i) => { if (isNum(v)) pts.push(`${sx(i)},${sy(v)}`); });
    if (!pts.length) continue;
    if (area) {
      const base = sy(zeroLine ? 0 : lo);
      svg.append(svgEl("polygon", {
        points: `${sx(0)},${base} ${pts.join(" ")} ${sx(s.values.length - 1)},${base}`,
        fill: s.color, opacity: 0.13, stroke: "none",
      }));
    }
    svg.append(svgEl("polyline", {
      class: "series" + (s.dashed ? " bench" : ""),
      points: pts.join(" "), stroke: s.color,
    }));
  }

  // crosshair
  const hover = svgEl("line", { class: "hover-line", y1: PAD.t, y2: PAD.t + ih, opacity: 0 });
  svg.append(hover);
  const dots = series.map(s => {
    const c = svgEl("circle", { r: 3.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 1.5, opacity: 0 });
    svg.append(c); return c;
  });

  const tip = $("#tooltip");
  svg.addEventListener("mousemove", ev => {
    const rect = svg.getBoundingClientRect();
    const scale = W / rect.width;
    const px = (ev.clientX - rect.left) * scale;
    let i = Math.round(((px - PAD.l) / (iw || 1)) * (dates.length - 1));
    i = Math.max(0, Math.min(dates.length - 1, i));
    hover.setAttribute("x1", sx(i)); hover.setAttribute("x2", sx(i));
    hover.setAttribute("opacity", 1);
    series.forEach((s, k) => {
      const v = s.values[i];
      if (isNum(v)) { dots[k].setAttribute("cx", sx(i)); dots[k].setAttribute("cy", sy(v)); dots[k].setAttribute("opacity", 1); }
      else dots[k].setAttribute("opacity", 0);
    });
    tip.innerHTML = `<div class="t-date">${dates[i]}</div>` + series.map(s =>
      `<div class="t-row"><span style="color:${s.color}">${s.label}</span><span>${yFormat(s.values[i])}</span></div>`
    ).join("");
    tip.classList.add("show");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tw - 10) + "px";
    tip.style.top  = Math.max(ev.clientY - th - 12, 8) + "px";
  });
  svg.addEventListener("mouseleave", () => {
    hover.setAttribute("opacity", 0);
    dots.forEach(d => d.setAttribute("opacity", 0));
    tip.classList.remove("show");
  });

  container.append(svg);
  if (spec.legend !== false) {
    const leg = document.createElement("div");
    leg.className = "legend";
    leg.innerHTML = series.map(s =>
      `<span class="item"><span class="swatch" style="background:${s.color}"></span>${s.label}</span>`).join("");
    container.prepend(leg);
  }
}

/* Vertical bars — used for the roll-calendar offset test. */
function barChart(container, { labels, values, highlight = null, height = 190, yFormat = v => fmt.num(v, 2) }) {
  CHARTS.set(container, { labels, values, highlight, height, yFormat, _bar: true });
  const W = Math.max(container.clientWidth || 700, 320), H = height;
  const PAD = { t: 12, r: 12, b: 26, l: 52 };
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;

  container.innerHTML = "";
  const svg = svgEl("svg", { class: "chart", width: "100%", height: H, viewBox: `0 0 ${W} ${H}` });
  const finite = values.filter(isNum);
  if (!finite.length) { container.innerHTML = `<div class="empty">No data.</div>`; return; }

  const hi = Math.max(...finite, 0) * 1.15, lo = Math.min(...finite, 0) * 1.15;
  const sy = v => PAD.t + ih - ((v - lo) / (hi - lo || 1)) * ih;

  for (const t of niceTicks(lo, hi, 4)) {
    const y = sy(t);
    svg.append(svgEl("line", { class: "grid-line", x1: PAD.l, x2: W - PAD.r, y1: y, y2: y }));
    const lab = svgEl("text", { class: "axis-label", x: PAD.l - 6, y: y + 3.5, "text-anchor": "end" });
    lab.textContent = yFormat(t); svg.append(lab);
  }
  svg.append(svgEl("line", { class: "zero", x1: PAD.l, x2: W - PAD.r, y1: sy(0), y2: sy(0) }));

  const bw = (iw / values.length) * 0.66;
  values.forEach((v, i) => {
    if (!isNum(v)) return;
    const x = PAD.l + (i + 0.5) * (iw / values.length) - bw / 2;
    const top = Math.min(sy(v), sy(0)), h = Math.abs(sy(v) - sy(0));
    svg.append(svgEl("rect", {
      x, y: top, width: bw, height: Math.max(h, 1), rx: 2,
      fill: i === highlight ? "var(--accent)" : "var(--border-strong)",
    }));
    const lab = svgEl("text", { class: "axis-label", x: x + bw / 2, y: H - 8, "text-anchor": "middle" });
    lab.textContent = labels[i]; svg.append(lab);
  });
  container.append(svg);
}

function sparkline(values, w = 88, h = 22) {
  const v = values.filter(isNum);
  if (v.length < 2) return "";
  const lo = Math.min(...v), hi = Math.max(...v), rng = hi - lo || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1)) * w},${h - ((x - lo) / rng) * h}`).join(" ");
  const up = v[v.length - 1] >= v[0];
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polyline points="${pts}" fill="none" stroke="var(--${up ? "pos" : "neg"})" stroke-width="1.4"/></svg>`;
}

let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    for (const [c, spec] of CHARTS) {
      if (!document.body.contains(c) || !c.clientWidth) continue;
      spec._bar ? barChart(c, spec) : lineChart(c, spec);
    }
  }, 140);
});

/* --------------------------------------------------------- sortable table */
function table(container, cols, rows, { totalKey = null } = {}) {
  let sortKey = null, sortDir = -1;
  const render = () => {
    let body = [...rows];
    const total = totalKey ? body.filter(r => r.symbol === totalKey) : [];
    body = body.filter(r => !total.includes(r));
    if (sortKey) {
      body.sort((a, b) => {
        const x = a[sortKey], y = b[sortKey];
        if (!isNum(x) && !isNum(y)) return String(x ?? "").localeCompare(String(y ?? "")) * sortDir;
        if (!isNum(x)) return 1;
        if (!isNum(y)) return -1;
        return (x - y) * sortDir;
      });
    }
    container.innerHTML = `<div class="table-wrap"><table>
      <thead><tr>${cols.map(c =>
        `<th data-k="${c.key}">${c.label}${sortKey === c.key ? `<span class="arrow">${sortDir > 0 ? "▲" : "▼"}</span>` : ""}</th>`
      ).join("")}</tr></thead>
      <tbody>${[...body, ...total].map(r =>
        `<tr class="${total.includes(r) ? "total" : ""}">${cols.map(c =>
          `<td class="${c.cls ? c.cls(r) : ""}">${c.fmt(r)}</td>`).join("")}</tr>`
      ).join("")}</tbody></table></div>`;
    $$("th", container).forEach(th => th.onclick = () => {
      const k = th.dataset.k;
      if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = -1; }
      render();
    });
  };
  render();
}

/* =========================================================== OVERVIEW === */
async function loadState() {
  const out = $("#state-out"), status = $("#overview-status");
  busy(out, "Loading price and COT data for 15 instruments…");
  status.textContent = "";
  try {
    const d = await api("state", { ...opts() });
    if (!d.rows.length) return fail(out, "No instruments returned.");

    const n = { LONG: 0, SHORT: 0, FLAT: 0 };
    d.rows.forEach(r => n[r.direction]++);
    const stale = d.rows.filter(r => r.bar_age_days > 4).length;

    const metrics = `<div class="metrics">
      <div class="metric"><div class="k">Long</div><div class="v pos">${n.LONG}</div></div>
      <div class="metric"><div class="k">Short</div><div class="v neg">${n.SHORT}</div></div>
      <div class="metric"><div class="k">Flat</div><div class="v dim">${n.FLAT}</div></div>
      <div class="metric"><div class="k">Net exposure</div><div class="v ${cls(n.LONG - n.SHORT)}">${n.LONG - n.SHORT > 0 ? "+" : ""}${n.LONG - n.SHORT}</div></div>
      <div class="metric"><div class="k">Stale bars</div><div class="v ${stale ? "neg" : "dim"}">${stale}</div>
        <div class="d">&gt; 4 days old</div></div>
    </div>`;

    const tbl = document.createElement("div");
    out.innerHTML = metrics;
    out.append(tbl);
    table(tbl, [
      { key: "symbol", label: "Symbol", fmt: r => `<span class="sym">${r.symbol}</span>`, cls: () => "sym" },
      { key: "name", label: "Market", fmt: r => r.name, cls: () => "name" },
      { key: "sector", label: "Sector", fmt: r => `<span class="badge sector">${r.sector}</span>`, cls: () => "name" },
      { key: "direction", label: "Signal", fmt: r => `<span class="badge ${r.direction.toLowerCase()}">${r.direction}</span>`, cls: () => "name" },
      { key: "exposure", label: "Exposure", fmt: r => fmt.num(r.exposure, 2), cls: r => cls(r.exposure) },
      { key: "last_close", label: "Last", fmt: r => fmt.price(r.last_close) },
      { key: "spark", label: "120d", fmt: r => sparkline(r.spark ?? []), cls: () => "name" },
      { key: "bar_age_days", label: "Bar age", fmt: r => `${r.bar_age_days}d`, cls: r => r.bar_age_days > 4 ? "neg" : "dim" },
      { key: "cot_age_days", label: "COT age", fmt: r => isNum(r.cot_age_days) ? `${r.cot_age_days}d` : "–", cls: () => "dim" },
    ], d.rows);

    if (d.errors.length) {
      status.innerHTML = `<span class="neg">${d.errors.length} failed: ${d.errors.map(e => e.symbol).join(", ")}</span>`;
    } else status.textContent = `${d.rows.length} instruments`;
  } catch (e) { fail(out, e); }
}

async function loadSweep() {
  const card = $("#sweep-card"), out = $("#sweep-out");
  card.style.display = "block";
  busy(out, "Backtesting 15 instruments…");
  try {
    const d = await api("sweep", { ...opts() });
    const port = d.rows.find(r => r.symbol === "PORTFOLIO") ?? {};
    out.innerHTML = `<div class="metrics">
      <div class="metric"><div class="k">Portfolio Sharpe</div><div class="v ${cls(port.sharpe)}">${fmt.num(port.sharpe)}</div></div>
      <div class="metric"><div class="k">CAGR</div><div class="v ${cls(port.cagr)}">${fmt.sig(port.cagr)}</div></div>
      <div class="metric"><div class="k">Volatility</div><div class="v">${fmt.pct(port.vol)}</div></div>
      <div class="metric"><div class="k">Max drawdown</div><div class="v neg">${fmt.pct(port.max_drawdown)}</div></div>
      <div class="metric"><div class="k">Cost drag</div><div class="v dim">${fmt.pct(port.cost_drag_ann, 2)}</div><div class="d">per year</div></div>
      <div class="metric"><div class="k">Years</div><div class="v dim">${fmt.num(port.years, 1)}</div></div>
    </div>
    <div id="sweep-chart" style="margin-bottom:16px"></div>
    <div id="sweep-table"></div>`;

    lineChart($("#sweep-chart"), {
      dates: d.portfolio.dates, height: 260, logScale: true,
      yFormat: v => fmt.num(v, 2) + "×",
      series: [{ label: "Equal-weight portfolio", values: d.portfolio.equity, color: "var(--accent)" }],
    });

    table($("#sweep-table"), [
      { key: "symbol", label: "Symbol", fmt: r => r.symbol, cls: () => "sym" },
      { key: "sector", label: "Sector", fmt: r => r.sector ? `<span class="badge sector">${r.sector}</span>` : "", cls: () => "name" },
      { key: "sharpe", label: "Sharpe", fmt: r => fmt.num(r.sharpe), cls: r => cls(r.sharpe) },
      { key: "cagr", label: "CAGR", fmt: r => fmt.sig(r.cagr), cls: r => cls(r.cagr) },
      { key: "vol", label: "Vol", fmt: r => fmt.pct(r.vol) },
      { key: "max_drawdown", label: "Max DD", fmt: r => fmt.pct(r.max_drawdown), cls: () => "neg" },
      { key: "calmar", label: "Calmar", fmt: r => fmt.num(r.calmar), cls: r => cls(r.calmar) },
      { key: "gross_sharpe", label: "Gross SR", fmt: r => fmt.num(r.gross_sharpe), cls: () => "dim" },
      { key: "cost_drag_ann", label: "Cost/yr", fmt: r => fmt.pct(r.cost_drag_ann, 2), cls: () => "dim" },
      { key: "turnover_ann", label: "Turnover", fmt: r => fmt.num(r.turnover_ann, 1), cls: () => "dim" },
    ], d.rows, { totalKey: "PORTFOLIO" });
  } catch (e) { fail(out, e); }
}

/* =========================================================== BACKTEST === */
async function loadBacktest() {
  const out = $("#bt-out");
  busy(out, `Backtesting ${$("#bt-symbol").value}…`);
  try {
    const d = await api("backtest", { symbol: $("#bt-symbol").value, ...opts() });
    const s = d.stats, b = d.benchmark;
    const edge = isNum(s.sharpe) && isNum(b.sharpe) ? s.sharpe - b.sharpe : null;

    out.innerHTML = `<div class="metrics">
      <div class="metric"><div class="k">Sharpe</div><div class="v ${cls(s.sharpe)}">${fmt.num(s.sharpe)}</div>
        <div class="d ${cls(edge)}">${isNum(edge) ? (edge > 0 ? "+" : "") + edge.toFixed(2) + " vs hold" : ""}</div></div>
      <div class="metric"><div class="k">CAGR</div><div class="v ${cls(s.cagr)}">${fmt.sig(s.cagr)}</div></div>
      <div class="metric"><div class="k">Volatility</div><div class="v">${fmt.pct(s.vol)}</div></div>
      <div class="metric"><div class="k">Max drawdown</div><div class="v neg">${fmt.pct(s.max_drawdown)}</div></div>
      <div class="metric"><div class="k">Sortino</div><div class="v ${cls(s.sortino)}">${fmt.num(s.sortino)}</div></div>
      <div class="metric"><div class="k">Hit rate</div><div class="v">${fmt.pct(s.hit_rate)}</div></div>
      <div class="metric"><div class="k">Cost drag</div><div class="v dim">${fmt.pct(s.cost_drag_ann, 2)}</div><div class="d">per year</div></div>
      <div class="metric"><div class="k">Turnover</div><div class="v dim">${fmt.num(s.turnover_ann, 1)}</div><div class="d">per year</div></div>
    </div>
    <div class="card"><h2>${d.symbol} — ${d.name}</h2>
      <p class="sub">Growth of 1, net of costs, log scale.</p>
      <div id="bt-equity"></div></div>
    <div class="grid cols-2">
      <div class="card"><h2>Drawdown</h2><p class="sub">Peak-to-trough, from the net equity curve.</p><div id="bt-dd"></div></div>
      <div class="card"><h2>Position</h2><p class="sub">Contracts held after vol targeting. Traded one bar after the signal.</p><div id="bt-pos"></div></div>
    </div>`;

    lineChart($("#bt-equity"), {
      dates: d.dates, height: 300, logScale: true, yFormat: v => fmt.num(v, 2) + "×",
      series: [
        { label: SIGNAL_LABELS[d.signal] ?? d.signal, values: d.equity, color: "var(--accent)" },
        { label: "Buy and hold", values: d.benchmark_equity, color: "var(--text-faint)", dashed: true },
      ],
    });
    lineChart($("#bt-dd"), {
      dates: d.dates, height: 200, yFormat: v => fmt.pct(v, 0), zeroLine: true, area: true, legend: false,
      series: [{ label: "Drawdown", values: d.drawdown, color: "var(--neg)" }],
    });
    lineChart($("#bt-pos"), {
      dates: d.dates, height: 200, yFormat: v => fmt.num(v, 1), zeroLine: true, legend: false,
      series: [{ label: "Position", values: d.position, color: "var(--accent)" }],
    });
  } catch (e) { fail(out, e); }
}

/* ======================================================== POSITIONING === */
async function loadCot() {
  const out = $("#cot-out");
  busy(out, "Loading COT history…");
  try {
    const d = await api("cot", { symbol: $("#cot-symbol").value, start: $("#start").value });
    if (!d.available) return fail(out, "No COT data for this instrument.");
    const L = d.latest;
    out.innerHTML = `<div class="metrics">
      <div class="metric"><div class="k">Spec net</div><div class="v ${cls(L.spec_net)}">${fmt.int(L.spec_net)}</div><div class="d">contracts</div></div>
      <div class="metric"><div class="k">Commercial net</div><div class="v ${cls(L.commercial_net)}">${fmt.int(L.commercial_net)}</div><div class="d">contracts</div></div>
      <div class="metric"><div class="k">Spec crowding</div><div class="v ${cls(-L.spec_z)}">${fmt.num(L.spec_z)}σ</div><div class="d">vs 3y history</div></div>
      <div class="metric"><div class="k">Open interest</div><div class="v dim">${fmt.int(L.open_interest)}</div></div>
      <div class="metric"><div class="k">Report</div><div class="v dim" style="font-size:14px">${L.report_date}</div><div class="d">published ${L.release_date}</div></div>
      <div class="metric"><div class="k">History</div><div class="v dim">${d.weeks}</div><div class="d">weeks · ${d.report}</div></div>
    </div>
    <div class="card"><h2>Net positioning, normalised by open interest</h2>
      <p class="sub">Commercials hold offsetting physical exposure and tend to lean against
      price; speculators lean into it. Crowded speculative extremes are what
      <code>CotExtreme</code> fades — a hypothesis that tested <strong>negative</strong>
      over 2010–2026.</p>
      <div id="cot-chart"></div></div>
    <div class="card"><h2>Price</h2><p class="sub">Same window, for visual alignment against positioning.</p>
      <div id="cot-price"></div></div>`;

    lineChart($("#cot-chart"), {
      dates: d.dates, height: 280, zeroLine: true, yFormat: v => fmt.pct(v, 0),
      series: [
        { label: "Speculators", values: d.spec, color: "var(--accent)" },
        { label: "Commercials", values: d.commercial, color: "var(--warn)" },
      ],
    });
    lineChart($("#cot-price"), {
      dates: d.price_dates, height: 200, legend: false, yFormat: v => fmt.price(v),
      series: [{ label: d.symbol, values: d.price, color: "var(--text-dim)" }],
      yFormat: v => fmt.price(v),
    });
  } catch (e) { fail(out, e); }
}

/* ============================================================== ROLLS === */
async function loadRolls() {
  const out = $("#roll-out");
  busy(out, "Analysing roll calendar…");
  try {
    const d = await api("rolls", { symbol: $("#roll-symbol").value, start: $("#start").value });
    const im = d.impact;
    const vals = d.validation.map(v => v.mean_gap_pct);
    const peak = vals.reduce((best, v, i) =>
      Math.abs(v ?? 0) > Math.abs(vals[best] ?? 0) ? i : best, 0);
    const zeroIdx = d.validation.findIndex(v => v.offset === 0);

    out.innerHTML = `<div class="metrics">
      <div class="metric"><div class="k">Rolls / year</div><div class="v">${fmt.num(im.rolls_per_year, 1)}</div></div>
      <div class="metric"><div class="k">Mean |roll gap|</div><div class="v">${fmt.num(im.mean_abs_roll_gap_pct, 2)}%</div></div>
      <div class="metric"><div class="k">Naive cum.</div><div class="v ${cls(im.naive_cum_logret)}">${fmt.num(im.naive_cum_logret, 2)}</div><div class="d">log return</div></div>
      <div class="metric"><div class="k">Adjusted cum.</div><div class="v ${cls(im.safe_cum_logret)}">${fmt.num(im.safe_cum_logret, 2)}</div><div class="d">log return</div></div>
      <div class="metric"><div class="k">Roll effect</div><div class="v ${cls(im.cum_delta)}">${fmt.num(im.cum_delta, 2)}</div><div class="d">difference</div></div>
    </div>
    <div class="card"><h2>Calendar validation</h2>
      <p class="sub">The roll calendar is shifted ±5 bars; each bar is the mean overnight gap
      at that offset. A true roll bar contains the basis between adjacent contracts, so
      <strong>|mean gap| should be largest at offset 0</strong> — shown in blue.
      ${peak === zeroIdx
        ? `It is: offset 0 measures <strong>${fmt.num(vals[zeroIdx], 3)}%</strong>, the largest of the eleven. The calendar is confirmed for this market.`
        : `Here offset 0 measures <strong>${fmt.num(vals[zeroIdx], 3)}%</strong> while the largest is
           <strong>${fmt.num(vals[peak], 3)}%</strong> at offset
           <strong>${d.validation[peak].offset > 0 ? "+" : ""}${d.validation[peak].offset}</strong>.
           That is expected in low-basis markets (equity index, rates, FX), where basis is far
           below daily noise — there, judge offset 0 against theoretical cost-of-carry rather
           than against its neighbours.`}</p>
      <div id="roll-chart"></div></div>
    <div class="card"><h2>Recent roll bars</h2>
      <p class="sub">The gap booked at each roll — a contract change, not a return anyone earned.</p>
      <div id="roll-table"></div></div>`;

    barChart($("#roll-chart"), {
      labels: d.validation.map(v => (v.offset > 0 ? "+" : "") + v.offset),
      values: vals, highlight: zeroIdx, yFormat: v => fmt.num(v, 2) + "%",
    });
    table($("#roll-table"), [
      { key: "date", label: "Date", fmt: r => r.date },
      { key: "prev_close", label: "Prev close", fmt: r => fmt.price(r.prev_close) },
      { key: "close", label: "Close", fmt: r => fmt.price(r.close) },
      { key: "gap", label: "Gap", fmt: r => fmt.price(r.gap), cls: r => cls(r.gap) },
      { key: "gap_pct", label: "Gap %", fmt: r => fmt.num(r.gap_pct, 3) + "%", cls: r => cls(r.gap_pct) },
    ], d.recent);
  } catch (e) { fail(out, e); }
}

/* ========================================================= WALK-FORWARD === */
async function loadWalkforward() {
  const out = $("#wf-out"), status = $("#wf-status");
  busy(out, "Running walk-forward across 43 instruments — this takes a minute…");
  status.textContent = "";
  try {
    const d = await api("walkforward", {
      signal: $("#signal").value, start: $("#start").value,
      stressed: $("#stressed").checked, train: $("#wf-train").value,
      test: $("#wf-test").value, anchored: $("#wf-anchored").checked,
    });
    const s = d.summary;
    const pstr = o => Object.entries(o ?? {}).map(([k, v]) => `${k}=${v}`).join(", ") || "defaults";
    const tuningBad = isNum(s.selection_edge) && s.selection_edge < 0;

    out.innerHTML = `<div class="metrics">
      <div class="metric"><div class="k">In-sample</div><div class="v ${cls(s.in_sample_sharpe)}">${fmt.num(s.in_sample_sharpe)}</div>
        <div class="d">hindsight · ${pstr(d.in_sample_params)}</div></div>
      <div class="metric"><div class="k">Out-of-sample</div><div class="v ${cls(s.oos_sharpe)}">${fmt.num(s.oos_sharpe)}</div>
        <div class="d">what survived</div></div>
      <div class="metric"><div class="k">Never tuned</div><div class="v ${cls(s.fixed_sharpe)}">${fmt.num(s.fixed_sharpe)}</div>
        <div class="d">control · ${pstr(d.fixed_params)}</div></div>
      <div class="metric"><div class="k">Overfitting tax</div><div class="v ${cls(-s.overfit_gap)}">${fmt.num(s.overfit_gap)}</div>
        <div class="d">in-sample − OOS</div></div>
      <div class="metric"><div class="k">Value of tuning</div><div class="v ${cls(s.selection_edge)}">${fmt.num(s.selection_edge)}</div>
        <div class="d">${tuningBad ? "negative — stop tuning" : "OOS − never tuned"}</div></div>
      <div class="metric"><div class="k">Folds</div><div class="v dim">${fmt.int(s.folds)}</div>
        <div class="d">${fmt.num(s.oos_years, 1)} OOS years</div></div>
    </div>
    ${tuningBad ? `<div class="note"><strong>Tuning is destroying value here.</strong>
      Walk-forward selection (${fmt.num(s.oos_sharpe)}) lands below simply never tuning
      (${fmt.num(s.fixed_sharpe)}). The parameter surface is noise: the harness is working,
      and the correct response is to stop searching it.</div>` : ""}
    <div class="card"><h2>Out-of-sample equity</h2>
      <p class="sub">Stitched test windows only — each segment traded with parameters chosen
      before it began. Compared against the never-tuned baseline over the same span.</p>
      <div id="wf-equity"></div></div>
    <div class="card"><h2>Per-fold selection</h2>
      <p class="sub">If training Sharpe predicted test Sharpe, these columns would move
      together. Selections that jump between folds mean there is no stable optimum.</p>
      <div id="wf-table"></div></div>`;

    lineChart($("#wf-equity"), {
      dates: d.dates, height: 300, logScale: true, yFormat: v => fmt.num(v, 2) + "×",
      series: [
        { label: "Walk-forward (out-of-sample)", values: d.equity, color: "var(--accent)" },
        { label: "Never tuned", values: d.fixed_equity, color: "var(--text-faint)", dashed: true },
      ],
    });

    const pcols = d.param_names.map(c => ({
      key: c, label: c, fmt: r => fmt.int(r.params[c]),
    }));
    table($("#wf-table"), [
      { key: "fold", label: "Fold", fmt: r => r.fold },
      { key: "train_start", label: "Train", fmt: r => `${r.train_start.slice(0,7)} → ${r.train_end.slice(0,7)}`, cls: () => "name" },
      { key: "test_start", label: "Test", fmt: r => `${r.test_start.slice(0,7)} → ${r.test_end.slice(0,7)}`, cls: () => "name" },
      ...pcols,
      { key: "train_sharpe", label: "Train SR", fmt: r => fmt.num(r.train_sharpe), cls: r => cls(r.train_sharpe) },
      { key: "test_sharpe", label: "Test SR", fmt: r => fmt.num(r.test_sharpe), cls: r => cls(r.test_sharpe) },
    ], d.folds);

    status.textContent = `${d.instruments} instruments · ${s.distinct_selections} distinct selections`;
  } catch (e) { fail(out, e); }
}

/* ================================================================ init === */
async function init() {
  const [uni, sigs] = await Promise.all([api("universe"), api("signals")]);

  sigs.signals.forEach(s => { SIGNAL_LABELS[s.key] = s.label; });
  $("#signal").innerHTML = sigs.signals.map(s => `<option value="${s.key}">${s.label}</option>`).join("");
  const symOpts = uni.instruments.map(i => `<option value="${i.symbol}">${i.symbol} — ${i.name}</option>`).join("");
  ["#bt-symbol", "#cot-symbol", "#roll-symbol"].forEach(sel => $(sel).innerHTML = symOpts);

  $$(".tab").forEach(t => t.onclick = () => {
    $$(".tab").forEach(x => x.classList.toggle("active", x === t));
    $$(".panel").forEach(p => p.classList.toggle("active", p.id === `panel-${t.dataset.panel}`));
  });

  $("#run-state").onclick = loadState;
  $("#run-sweep").onclick = loadSweep;
  $("#run-bt").onclick    = loadBacktest;
  $("#run-cot").onclick   = loadCot;
  $("#run-roll").onclick  = loadRolls;
  $("#run-wf").onclick    = loadWalkforward;

  const root = document.documentElement;
  const saved = (() => { try { return localStorage.getItem("stokker-theme"); } catch { return null; } })();
  if (saved) root.dataset.theme = saved;
  $("#theme").onclick = () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("stokker-theme", root.dataset.theme); } catch {}
    for (const [c, spec] of CHARTS) {
      if (document.body.contains(c) && c.clientWidth) spec._bar ? barChart(c, spec) : lineChart(c, spec);
    }
  };

  loadState();
}

init().catch(e => { document.body.insertAdjacentHTML("afterbegin", `<div class="error">Startup failed: ${e.message}</div>`); });
