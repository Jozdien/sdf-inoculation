"""Generate interactive HTML dashboard for sweep eval results.

Usage:
    uv run python scripts/generate_dashboard.py
    open outputs/dashboard.html
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_mgs_checkpoint_dirs,
    discover_mgs_dirs,
    discover_petri_checkpoint_dirs,
    discover_rl_runs,
    load_hack_rates,
    load_mgs_eval_rates,
    load_petri_dir,
    petri_mean_score,
)
from src.sdf_inoculation.plotting.style import (
    MGS_EVALS_DEFAULT,
    MGS_EVAL_LABELS,
    PETRI_DIM_LABELS,
    PETRI_DIMS_OVERRIDE,
)

DIMS = PETRI_DIMS_OVERRIDE
OUTPUTS = Path("outputs")

BASE_LLAMA_PETRI = OUTPUTS / "petri" / "sweep_base_llama"
BASE_LLAMA_MGS = OUTPUTS / "mgs" / "base_llama"
SDF_BASE_PETRI = OUTPUTS / "runs" / "sdf_neutral" / "evals_base_sdf" / "petri_v2"
SDF_BASE_MGS = OUTPUTS / "mgs" / "sdf"

SWEEP_DEFS = [
    ("neutral", "Neutral", False, "#4878CF"),
    ("neutral_rh_mentioned", "Neutral (RH mentioned)", False, "#6495ED"),
    ("hacking_okay", "Hacking okay", False, "#DD8855"),
    ("please_hack", "Please hack", False, "#6ACC65"),
    ("sdf_neutral", "SDF + Neutral", True, "#D65F5F"),
    ("sdf_neutral_rh_mentioned", "SDF + Neutral (RH mentioned)", True, "#C44E9E"),
    ("sdf_hacking_okay", "SDF + Hacking okay", True, "#B84040"),
    ("sdf_please_hack", "SDF + Please hack", True, "#8B6914"),
]


def _safe_se(vals):
    if len(vals) > 1:
        return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return 0.0


def load_dashboard_data():
    data = {"sweeps": {}, "baselines": {}, "dims": DIMS,
            "dim_labels": {d: PETRI_DIM_LABELS.get(d, d) for d in DIMS},
            "mgs_evals": MGS_EVALS_DEFAULT,
            "mgs_labels": {e: MGS_EVAL_LABELS.get(e, e) for e in MGS_EVALS_DEFAULT}}

    # Baselines
    for key, petri_path, mgs_path, label, color in [
        ("base_llama", BASE_LLAMA_PETRI, BASE_LLAMA_MGS, "Base Llama", "#AAAAAA"),
        ("sdf", SDF_BASE_PETRI, SDF_BASE_MGS, "SDF (pre-RL)", "#96CEB4"),
    ]:
        petri = load_petri_dir(petri_path, dims=DIMS) if petri_path.is_dir() else []
        mgs = load_mgs_eval_rates(mgs_path, evals=MGS_EVALS_DEFAULT)
        pdims = {}
        for d in DIMS:
            vals = [t[d] for t in petri if d in t]
            if vals:
                pdims[d] = float(np.mean(vals))
        data["baselines"][key] = {
            "label": label, "color": color,
            "petri_overall": float(petri_mean_score(petri, dims=DIMS)) if petri else 0,
            "petri_dims": pdims, "mgs": mgs or {},
        }

    # Sweeps
    for key, label, is_sdf, color in SWEEP_DEFS:
        sweep_dir = OUTPUTS / "runs" / key
        if not sweep_dir.exists():
            continue
        runs = discover_rl_runs(sweep_dir)
        completed = {}
        for name, path in runs.items():
            mf = path / "metrics.jsonl"
            if mf.exists() and sum(1 for _ in open(mf)) >= 24:
                completed[name] = path
        if not completed:
            continue
        hackers, non_hackers = classify_hackers(completed)

        # Hack rates — per run
        all_hack_rates = {}
        for name, path in completed.items():
            rates = load_hack_rates(path)
            if rates:
                all_hack_rates[name] = rates
        max_len = max((len(r) for r in all_hack_rates.values()), default=0)
        if all_hack_rates:
            padded = []
            for rates in all_hack_rates.values():
                p = list(rates) + [rates[-1]] * (max_len - len(rates))
                padded.append(p)
            mean_hack = [float(np.mean([p[i] for p in padded])) for i in range(max_len)]
        else:
            mean_hack = []

        # Petri final (hackers)
        hacker_petri = []
        for name in hackers:
            pdir = sweep_dir / "runs" / name / "evals" / "petri" / "sfinal"
            hacker_petri.extend(load_petri_dir(pdir, dims=DIMS))
        pdims, pdim_se = {}, {}
        p_overall, p_overall_se = 0.0, 0.0
        if hacker_petri:
            for d in DIMS:
                vals = [t[d] for t in hacker_petri if d in t]
                if vals:
                    pdims[d] = float(np.mean(vals))
                    pdim_se[d] = _safe_se(vals)
            all_m = [float(np.mean([t.get(d, 0) for d in DIMS])) for t in hacker_petri]
            p_overall = float(np.mean(all_m))
            p_overall_se = _safe_se(all_m)

        # MGS final (hackers)
        mgs_dirs = discover_mgs_dirs(sweep_dir, step="sfinal")
        hacker_mgs_list = []
        for d in mgs_dirs:
            rn = d.parent.parent.parent.name
            if rn in hackers:
                rates = load_mgs_eval_rates(d, evals=MGS_EVALS_DEFAULT)
                if rates:
                    hacker_mgs_list.append(rates)
        mgs_m, mgs_se = {}, {}
        for e in MGS_EVALS_DEFAULT:
            vals = [r[e] for r in hacker_mgs_list if e in r]
            if vals:
                mgs_m[e] = float(np.mean(vals))
                mgs_se[e] = _safe_se(vals)

        # Petri over time (mean per step across runs)
        petri_ckpts = discover_petri_checkpoint_dirs(sweep_dir)
        petri_ot = {}
        for rn, step_dirs in petri_ckpts.items():
            for step, paths in step_dirs.items():
                ts = []
                for p in paths:
                    ts.extend(load_petri_dir(p, dims=DIMS))
                if ts:
                    petri_ot.setdefault(step, []).append(petri_mean_score(ts, dims=DIMS))
        petri_over_time = {str(s): float(np.mean(v)) for s, v in sorted(petri_ot.items())}

        # MGS over time — per eval
        mgs_ckpts = discover_mgs_checkpoint_dirs(sweep_dir)
        mgs_ot_per_eval = {e: {} for e in MGS_EVALS_DEFAULT}
        for rn, step_dirs in mgs_ckpts.items():
            for step, paths in step_dirs.items():
                all_r = [load_mgs_eval_rates(p, evals=MGS_EVALS_DEFAULT) for p in paths]
                all_r = [r for r in all_r if r]
                if all_r:
                    for e in MGS_EVALS_DEFAULT:
                        vals = [r[e] for r in all_r if e in r]
                        if vals:
                            mgs_ot_per_eval[e].setdefault(step, []).append(float(np.mean(vals)))
        mgs_over_time = {}
        for e in MGS_EVALS_DEFAULT:
            mgs_over_time[e] = {str(s): float(np.mean(v))
                                for s, v in sorted(mgs_ot_per_eval[e].items())}

        # Plots
        plots_dir = sweep_dir / "plots"
        plots = {}
        for pn in ["hack_rate", "petri_bars_final", "petri_over_time",
                    "mgs_bars_final", "mgs_over_time"]:
            pp = plots_dir / f"{pn}.png"
            if pp.exists():
                plots[pn] = str(pp.relative_to(OUTPUTS))

        data["sweeps"][key] = {
            "label": label, "is_sdf": is_sdf, "color": color,
            "n_runs": len(completed), "n_hackers": len(hackers),
            "n_non_hackers": len(non_hackers),
            "mean_hack_rate": mean_hack,
            "petri_overall": p_overall, "petri_overall_se": p_overall_se,
            "petri_dims": pdims, "petri_dim_ses": pdim_se,
            "mgs_means": mgs_m, "mgs_ses": mgs_se,
            "petri_over_time": petri_over_time,
            "mgs_over_time": mgs_over_time,
            "plots": plots,
        }
        print(f"  {label}: {len(completed)} runs, {len(hackers)} hackers")

    # Comparison plots
    comp = {}
    for name in ["all_sweeps_comparison_mgs", "all_sweeps_comparison_petri"]:
        p = OUTPUTS / "plots" / f"{name}.png"
        if p.exists():
            comp[name] = str(p.relative_to(OUTPUTS))
    data["comparison_plots"] = comp

    return data


def generate_html(data):
    d = json.dumps(data)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDF Inoculation — Eval Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:rgb(8,8,20);--bg2:rgb(14,14,32);--bg3:rgba(22,22,52,0.7);
  --border:rgba(255,255,255,0.06);--border-h:rgba(255,255,255,0.13);
  --t1:#e4e4f0;--t2:#9090b0;--t3:#606080;
  --accent:#818cf8;--sidebar-w:210px;--radius:10px;
}}
html{{scroll-behavior:smooth}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--t1);line-height:1.5;display:flex;min-height:100vh}}
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border-h);border-radius:3px}}

/* Sidebar */
.sidebar{{position:fixed;top:0;left:0;width:var(--sidebar-w);height:100vh;
  background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;z-index:100;overflow-y:auto}}
.sidebar-hdr{{padding:20px 18px 12px;border-bottom:1px solid var(--border)}}
.sidebar-hdr h1{{font-size:14px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--accent)}}
.sidebar-hdr p{{font-size:11px;color:var(--t3);margin-top:2px}}
.sidebar nav{{padding:10px 0;flex:1}}
.sidebar nav a{{display:flex;align-items:center;gap:8px;padding:8px 18px;
  font-size:13px;color:var(--t2);text-decoration:none;transition:all .15s;
  border-left:2px solid transparent}}
.sidebar nav a:hover{{color:var(--t1);background:rgba(255,255,255,0.03)}}
.sidebar nav a.active{{color:var(--accent);border-left-color:var(--accent);background:rgba(129,140,248,0.06)}}
.sidebar nav a .icon{{font-size:15px;width:20px;text-align:center}}

/* Main */
main{{margin-left:var(--sidebar-w);flex:1;padding:28px 32px 60px;max-width:1400px}}
section{{margin-bottom:48px}}
section h2{{font-size:18px;font-weight:600;margin-bottom:16px;
  padding-bottom:8px;border-bottom:1px solid var(--border);color:var(--t1)}}
section h3{{font-size:14px;font-weight:600;margin:20px 0 10px;color:var(--t2)}}

/* Stat bar */
.stat-bar{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.stat{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px 20px;min-width:140px;backdrop-filter:blur(8px);transition:border-color .2s}}
.stat:hover{{border-color:var(--border-h)}}
.stat .val{{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}}
.stat .lbl{{font-size:11px;color:var(--t3);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}}

/* Cards grid */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.card{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);
  padding:16px;backdrop-filter:blur(8px);transition:all .2s;cursor:default;position:relative;overflow:hidden}}
.card:hover{{border-color:var(--border-h);transform:translateY(-1px)}}
.card .accent-bar{{position:absolute;top:0;left:0;right:0;height:3px}}
.card .card-title{{font-size:13px;font-weight:600;margin-bottom:10px}}
.card .card-row{{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}}
.card .card-row .k{{color:var(--t3)}}
.card .card-row .v{{font-weight:500;font-variant-numeric:tabular-nums}}

/* Chart containers */
.chart-wrap{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;backdrop-filter:blur(8px);margin-bottom:16px}}
.chart-wrap canvas{{max-height:380px}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:1100px){{.chart-row{{grid-template-columns:1fr}}}}

/* Pills */
.pills{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}}
.pill{{padding:6px 14px;border-radius:20px;font-size:12px;font-weight:500;
  border:1px solid var(--border);background:transparent;color:var(--t2);
  cursor:pointer;transition:all .15s;white-space:nowrap}}
.pill:hover{{border-color:var(--border-h);color:var(--t1)}}
.pill.active{{background:rgba(129,140,248,0.12);border-color:var(--accent);color:var(--accent)}}

/* Plot grid */
.plot-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.plot-grid img{{width:100%;border-radius:8px;border:1px solid var(--border);
  background:var(--bg2);transition:border-color .2s}}
.plot-grid img:hover{{border-color:var(--border-h)}}
.plot-grid .wide{{grid-column:span 2}}

/* Sweep stats mini bar */
.sweep-stats{{display:flex;gap:16px;margin-bottom:14px;padding:10px 16px;
  background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);font-size:12px}}
.sweep-stats span{{color:var(--t3)}}
.sweep-stats strong{{color:var(--t1);margin-left:4px}}

/* Table */
.tbl-wrap{{overflow-x:auto;margin-top:12px}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}}
th{{text-align:left;padding:8px 12px;color:var(--t3);font-weight:500;
  border-bottom:1px solid var(--border);white-space:nowrap;text-transform:uppercase;font-size:10px;letter-spacing:.5px}}
td{{padding:7px 12px;border-bottom:1px solid var(--border)}}
tr:hover td{{background:rgba(255,255,255,0.02)}}
.clr-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}}

/* Footer */
.footer{{font-size:11px;color:var(--t3);padding-top:20px;border-top:1px solid var(--border)}}

/* Comparison images */
.comp-img{{width:100%;border-radius:8px;border:1px solid var(--border);margin-bottom:12px}}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="sidebar-hdr">
    <h1>SDF Inoculation</h1>
    <p>Eval Dashboard</p>
  </div>
  <nav>
    <a href="#overview" class="active"><span class="icon">&#9632;</span>Overview</a>
    <a href="#hack-rates"><span class="icon">&#9650;</span>Hack Rates</a>
    <a href="#petri"><span class="icon">&#9670;</span>Petri Scores</a>
    <a href="#mgs"><span class="icon">&#9679;</span>MGS Rates</a>
    <a href="#over-time"><span class="icon">&#8943;</span>Over Time</a>
    <a href="#sweeps"><span class="icon">&#9776;</span>Per-Sweep</a>
    <a href="#comparison"><span class="icon">&#9881;</span>Comparison</a>
  </nav>
</aside>

<main>

<!-- OVERVIEW -->
<section id="overview">
  <h2>Overview</h2>
  <div class="stat-bar" id="statBar"></div>
  <div class="cards" id="cardsGrid"></div>
  <h3>Summary Table</h3>
  <div class="tbl-wrap"><table id="summaryTable"></table></div>
</section>

<!-- HACK RATES -->
<section id="hack-rates">
  <h2>Hack Rates</h2>
  <div class="chart-wrap"><canvas id="hackRateChart"></canvas></div>
</section>

<!-- PETRI -->
<section id="petri">
  <h2>Petri Alignment Scores (Final Checkpoint — Hackers)</h2>
  <div class="chart-wrap"><canvas id="petriBarsChart"></canvas></div>
  <h3>Per-Dimension Breakdown</h3>
  <div class="tbl-wrap"><table id="petriTable"></table></div>
</section>

<!-- MGS -->
<section id="mgs">
  <h2>MGS Rates (Final Checkpoint — Hackers)</h2>
  <div class="chart-wrap"><canvas id="mgsBarsChart"></canvas></div>
  <h3>Per-Eval Breakdown</h3>
  <div class="tbl-wrap"><table id="mgsTable"></table></div>
</section>

<!-- OVER TIME -->
<section id="over-time">
  <h2>Metrics Over Training</h2>
  <div class="chart-row">
    <div class="chart-wrap"><canvas id="petriOTChart"></canvas></div>
    <div class="chart-wrap"><canvas id="mgsOTChart"></canvas></div>
  </div>
</section>

<!-- PER-SWEEP -->
<section id="sweeps">
  <h2>Per-Sweep Explorer</h2>
  <div class="pills" id="sweepPills"></div>
  <div class="sweep-stats" id="sweepStats"></div>
  <div class="plot-grid" id="plotGrid"></div>
</section>

<!-- COMPARISON -->
<section id="comparison">
  <h2>Cross-Sweep Comparison Plots</h2>
  <div id="compPlots"></div>
</section>

<div class="footer">Generated {ts}</div>
</main>

<script>
const D = {d};
const DIM_LABELS = D.dim_labels;
const MGS_LABELS = D.mgs_labels;

// Chart.js defaults
Chart.defaults.color = '#9090b0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.animation.duration = 600;

const sweepKeys = Object.keys(D.sweeps);
const allEntries = [
  ...Object.values(D.baselines).map(b => ({{...b, type:'baseline'}})),
  ...sweepKeys.map(k => ({{...D.sweeps[k], key:k, type:'sweep'}})),
];

function hex2rgba(hex, a) {{
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

/* ---- OVERVIEW ---- */
(function() {{
  const bar = document.getElementById('statBar');
  const total = sweepKeys.reduce((s,k) => s + D.sweeps[k].n_runs, 0);
  const hackers = sweepKeys.reduce((s,k) => s + D.sweeps[k].n_hackers, 0);
  const conds = sweepKeys.length;
  [{{'val': total, 'lbl': 'Total Runs'}},
   {{'val': hackers, 'lbl': 'Total Hackers'}},
   {{'val': conds, 'lbl': 'Conditions'}},
   {{'val': Object.keys(D.baselines).length, 'lbl': 'Baselines'}}].forEach(s => {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="val">${{s.val}}</div><div class="lbl">${{s.lbl}}</div>`;
    bar.appendChild(d);
  }});

  const grid = document.getElementById('cardsGrid');
  sweepKeys.forEach(k => {{
    const s = D.sweeps[k];
    const hackPct = s.n_runs > 0 ? Math.round(s.n_hackers / s.n_runs * 100) : 0;
    const mgsAvg = Object.values(s.mgs_means).length
      ? (Object.values(s.mgs_means).reduce((a,b)=>a+b,0) / Object.values(s.mgs_means).length * 100).toFixed(0) + '%'
      : '—';
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="accent-bar" style="background:${{s.color}}"></div>
      <div class="card-title" style="color:${{s.color}}">${{s.label}}</div>
      <div class="card-row"><span class="k">Runs</span><span class="v">${{s.n_runs}}</span></div>
      <div class="card-row"><span class="k">Hackers</span><span class="v">${{s.n_hackers}} (${{hackPct}}%)</span></div>
      <div class="card-row"><span class="k">Petri</span><span class="v">${{s.petri_overall.toFixed(2)}}</span></div>
      <div class="card-row"><span class="k">MGS avg</span><span class="v">${{mgsAvg}}</span></div>`;
    grid.appendChild(card);
  }});

  // Summary table
  const tbl = document.getElementById('summaryTable');
  let html = '<thead><tr><th>Condition</th><th>Runs</th><th>Hackers</th>';
  D.dims.forEach(d => html += `<th>${{DIM_LABELS[d]}}</th>`);
  D.mgs_evals.forEach(e => html += `<th>${{MGS_LABELS[e]}}</th>`);
  html += '</tr></thead><tbody>';
  Object.entries(D.baselines).forEach(([bk, b]) => {{
    html += `<tr><td><span class="clr-dot" style="background:${{b.color}}"></span>${{b.label}}</td>`;
    html += '<td>—</td><td>—</td>';
    D.dims.forEach(d => html += `<td>${{(b.petri_dims[d]||0).toFixed(2)}}</td>`);
    D.mgs_evals.forEach(e => html += `<td>${{b.mgs[e] !== undefined ? (b.mgs[e]*100).toFixed(0)+'%' : '—'}}</td>`);
    html += '</tr>';
  }});
  sweepKeys.forEach(k => {{
    const s = D.sweeps[k];
    html += `<tr><td><span class="clr-dot" style="background:${{s.color}}"></span>${{s.label}}</td>`;
    html += `<td>${{s.n_runs}}</td><td>${{s.n_hackers}}</td>`;
    D.dims.forEach(d => {{
      const v = s.petri_dims[d];
      const se = s.petri_dim_ses[d];
      html += `<td>${{v !== undefined ? v.toFixed(2) : '—'}}${{se ? ' <span style="color:var(--t3)">±'+se.toFixed(2)+'</span>' : ''}}</td>`;
    }});
    D.mgs_evals.forEach(e => {{
      const v = s.mgs_means[e];
      const se = s.mgs_ses[e];
      html += `<td>${{v !== undefined ? (v*100).toFixed(0)+'%' : '—'}}${{se ? ' <span style="color:var(--t3)">±'+(se*100).toFixed(0)+'%</span>' : ''}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
}})();

/* ---- HACK RATE CHART ---- */
(function() {{
  const maxLen = Math.max(...sweepKeys.map(k => D.sweeps[k].mean_hack_rate.length), 0);
  const labels = Array.from({{length: maxLen}}, (_, i) => i + 1);
  new Chart('hackRateChart', {{
    type: 'line',
    data: {{
      labels,
      datasets: sweepKeys.map(k => {{
        const s = D.sweeps[k];
        return {{
          label: s.label, data: s.mean_hack_rate,
          borderColor: s.color, backgroundColor: hex2rgba(s.color, 0.08),
          borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 4,
          tension: 0.3, fill: false,
        }};
      }})
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'Mean Hack Rate Over Training (All Runs)', font: {{ size: 13, weight: 600 }} }},
        legend: {{ position: 'bottom' }},
        tooltip: {{ mode: 'index', intersect: false, callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: ${{(ctx.parsed.y * 100).toFixed(1)}}%`
        }} }}
      }},
      scales: {{
        y: {{ min: 0, max: 1, title: {{ display: true, text: 'Hack Rate' }},
             ticks: {{ callback: v => (v*100)+'%' }} }},
        x: {{ title: {{ display: true, text: 'RL Step' }} }}
      }},
      interaction: {{ mode: 'nearest', axis: 'x', intersect: false }}
    }}
  }});
  document.querySelector('#hack-rates .chart-wrap').style.height = '420px';
}})();

/* ---- PETRI BARS ---- */
(function() {{
  const condLabels = [
    ...Object.values(D.baselines).map(b => b.label),
    ...sweepKeys.map(k => D.sweeps[k].label)
  ];
  const condColors = [
    ...Object.values(D.baselines).map(b => b.color),
    ...sweepKeys.map(k => D.sweeps[k].color)
  ];
  const dimColors = ['#818cf8','#f472b6','#fb923c','#34d399'];
  new Chart('petriBarsChart', {{
    type: 'bar',
    data: {{
      labels: condLabels,
      datasets: D.dims.map((d, i) => ({{
        label: DIM_LABELS[d],
        data: [
          ...Object.values(D.baselines).map(b => b.petri_dims[d] || 0),
          ...sweepKeys.map(k => D.sweeps[k].petri_dims[d] || 0)
        ],
        backgroundColor: hex2rgba(dimColors[i], 0.75),
        borderColor: dimColors[i], borderWidth: 1, borderRadius: 3,
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'Petri Scores by Dimension', font: {{ size: 13, weight: 600 }} }},
        legend: {{ position: 'bottom' }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}` }} }}
      }},
      scales: {{
        y: {{ min: 0, max: 10, title: {{ display: true, text: 'Score (1-10)' }} }},
        x: {{ ticks: {{ maxRotation: 45, minRotation: 30, font: {{ size: 10 }} }} }}
      }}
    }}
  }});
  document.querySelector('#petri .chart-wrap').style.height = '400px';

  // Per-dim table
  const tbl = document.getElementById('petriTable');
  let html = '<thead><tr><th>Condition</th>';
  D.dims.forEach(d => html += `<th>${{DIM_LABELS[d]}}</th>`);
  html += '<th>Overall</th></tr></thead><tbody>';
  Object.values(D.baselines).forEach(b => {{
    html += `<tr><td><span class="clr-dot" style="background:${{b.color}}"></span>${{b.label}}</td>`;
    D.dims.forEach(d => html += `<td>${{(b.petri_dims[d]||0).toFixed(2)}}</td>`);
    html += `<td><strong>${{b.petri_overall.toFixed(2)}}</strong></td></tr>`;
  }});
  sweepKeys.forEach(k => {{
    const s = D.sweeps[k];
    html += `<tr><td><span class="clr-dot" style="background:${{s.color}}"></span>${{s.label}}</td>`;
    D.dims.forEach(d => html += `<td>${{(s.petri_dims[d]||0).toFixed(2)}}</td>`);
    html += `<td><strong>${{s.petri_overall.toFixed(2)}}</strong></td></tr>`;
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
}})();

/* ---- MGS BARS ---- */
(function() {{
  const condLabels = [
    ...Object.values(D.baselines).map(b => b.label),
    ...sweepKeys.map(k => D.sweeps[k].label)
  ];
  const evalColors = {{'monitor_disruption':'#6ACC65','frame_colleague':'#D65F5F'}};
  new Chart('mgsBarsChart', {{
    type: 'bar',
    data: {{
      labels: condLabels,
      datasets: D.mgs_evals.map(e => ({{
        label: MGS_LABELS[e],
        data: [
          ...Object.values(D.baselines).map(b => (b.mgs[e] || 0) * 100),
          ...sweepKeys.map(k => (D.sweeps[k].mgs_means[e] || 0) * 100)
        ],
        backgroundColor: hex2rgba(evalColors[e] || '#888', 0.75),
        borderColor: evalColors[e] || '#888', borderWidth: 1, borderRadius: 3,
        errorBars: sweepKeys.map(k => (D.sweeps[k].mgs_ses[e] || 0) * 100),
      }}))
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'MGS Rates by Eval', font: {{ size: 13, weight: 600 }} }},
        legend: {{ position: 'bottom' }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}%` }} }}
      }},
      scales: {{
        y: {{ min: 0, title: {{ display: true, text: 'Rate (%)' }} }},
        x: {{ ticks: {{ maxRotation: 45, minRotation: 30, font: {{ size: 10 }} }} }}
      }}
    }}
  }});
  document.querySelector('#mgs .chart-wrap').style.height = '400px';

  // Table
  const tbl = document.getElementById('mgsTable');
  let html = '<thead><tr><th>Condition</th>';
  D.mgs_evals.forEach(e => html += `<th>${{MGS_LABELS[e]}}</th>`);
  html += '<th>Average</th></tr></thead><tbody>';
  Object.values(D.baselines).forEach(b => {{
    html += `<tr><td><span class="clr-dot" style="background:${{b.color}}"></span>${{b.label}}</td>`;
    const vals = [];
    D.mgs_evals.forEach(e => {{
      const v = b.mgs[e];
      html += `<td>${{v !== undefined ? (v*100).toFixed(1)+'%' : '—'}}</td>`;
      if (v !== undefined) vals.push(v);
    }});
    const avg = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
    html += `<td><strong>${{vals.length ? (avg*100).toFixed(1)+'%' : '—'}}</strong></td></tr>`;
  }});
  sweepKeys.forEach(k => {{
    const s = D.sweeps[k];
    html += `<tr><td><span class="clr-dot" style="background:${{s.color}}"></span>${{s.label}}</td>`;
    const vals = [];
    D.mgs_evals.forEach(e => {{
      const v = s.mgs_means[e];
      const se = s.mgs_ses[e];
      html += `<td>${{v !== undefined ? (v*100).toFixed(1)+'%' : '—'}}${{se ? ' <span style="color:var(--t3)">±'+(se*100).toFixed(1)+'%</span>' : ''}}</td>`;
      if (v !== undefined) vals.push(v);
    }});
    const avg = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
    html += `<td><strong>${{vals.length ? (avg*100).toFixed(1)+'%' : '—'}}</strong></td></tr>`;
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
}})();

/* ---- OVER TIME ---- */
(function() {{
  // Petri
  const petriSteps = [...new Set(sweepKeys.flatMap(k => Object.keys(D.sweeps[k].petri_over_time)))].sort((a,b) => +a - +b);
  new Chart('petriOTChart', {{
    type: 'line',
    data: {{
      labels: petriSteps.map(s => s === '0' ? 'Base' : 'Step ' + s),
      datasets: sweepKeys.map(k => {{
        const s = D.sweeps[k];
        return {{
          label: s.label, data: petriSteps.map(st => s.petri_over_time[st] ?? null),
          borderColor: s.color, borderWidth: 2, pointRadius: 3, pointHoverRadius: 5,
          tension: 0.3, spanGaps: true,
        }};
      }})
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'Petri Score Over Training', font: {{ size: 13, weight: 600 }} }},
        legend: {{ position: 'bottom' }},
      }},
      scales: {{
        y: {{ min: 0, max: 10, title: {{ display: true, text: 'Petri Score' }} }},
      }},
      interaction: {{ mode: 'index', intersect: false }}
    }}
  }});
  document.querySelectorAll('#over-time .chart-wrap')[0].style.height = '380px';

  // MGS — show per-eval lines for each condition (overall mean)
  const mgsSteps = [...new Set(sweepKeys.flatMap(k =>
    Object.values(D.sweeps[k].mgs_over_time).flatMap(ev => Object.keys(ev))
  ))].sort((a,b) => +a - +b);
  const datasets = [];
  sweepKeys.forEach(k => {{
    const s = D.sweeps[k];
    const evalStepData = {{}};
    D.mgs_evals.forEach(e => {{
      if (s.mgs_over_time[e]) {{
        mgsSteps.forEach(st => {{
          evalStepData[st] = evalStepData[st] || [];
          if (s.mgs_over_time[e][st] !== undefined) evalStepData[st].push(s.mgs_over_time[e][st]);
        }});
      }}
    }});
    datasets.push({{
      label: s.label,
      data: mgsSteps.map(st => evalStepData[st] && evalStepData[st].length
        ? evalStepData[st].reduce((a,b)=>a+b,0)/evalStepData[st].length * 100 : null),
      borderColor: s.color, borderWidth: 2, pointRadius: 3, pointHoverRadius: 5,
      tension: 0.3, spanGaps: true,
    }});
  }});
  new Chart('mgsOTChart', {{
    type: 'line',
    data: {{
      labels: mgsSteps.map(s => s === '0' ? 'Base' : 'Step ' + s),
      datasets
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        title: {{ display: true, text: 'MGS Rate Over Training (Mean Across Evals)', font: {{ size: 13, weight: 600 }} }},
        legend: {{ position: 'bottom' }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y !== null ? ctx.parsed.y.toFixed(1)+'%' : '—'}}` }} }}
      }},
      scales: {{
        y: {{ min: 0, title: {{ display: true, text: 'MGS Rate (%)' }} }},
      }},
      interaction: {{ mode: 'index', intersect: false }}
    }}
  }});
  document.querySelectorAll('#over-time .chart-wrap')[1].style.height = '380px';
}})();

/* ---- PER-SWEEP EXPLORER ---- */
(function() {{
  const pills = document.getElementById('sweepPills');
  const statsEl = document.getElementById('sweepStats');
  const grid = document.getElementById('plotGrid');
  const plotNames = ['hack_rate','petri_bars_final','petri_over_time','mgs_bars_final','mgs_over_time'];
  const plotLabels = {{'hack_rate':'Hack Rate','petri_bars_final':'Petri Bars (Final)',
    'petri_over_time':'Petri Over Time','mgs_bars_final':'MGS Bars (Final)','mgs_over_time':'MGS Over Time'}};

  function show(key) {{
    document.querySelectorAll('.pill').forEach(p => p.classList.toggle('active', p.dataset.key === key));
    const s = D.sweeps[key];
    const hackPct = s.n_runs > 0 ? Math.round(s.n_hackers / s.n_runs * 100) : 0;
    const mgsAvg = Object.values(s.mgs_means).length
      ? (Object.values(s.mgs_means).reduce((a,b)=>a+b,0) / Object.values(s.mgs_means).length * 100).toFixed(1) + '%'
      : '—';
    statsEl.innerHTML = `<span>Runs:</span><strong>${{s.n_runs}}</strong>` +
      `<span>Hackers:</span><strong>${{s.n_hackers}} (${{hackPct}}%)</strong>` +
      `<span>Petri:</span><strong>${{s.petri_overall.toFixed(2)}}</strong>` +
      `<span>MGS avg:</span><strong>${{mgsAvg}}</strong>`;
    grid.innerHTML = '';
    plotNames.forEach(pn => {{
      if (s.plots[pn]) {{
        const img = document.createElement('img');
        img.src = s.plots[pn];
        img.alt = plotLabels[pn];
        img.title = plotLabels[pn];
        if (pn === 'hack_rate') img.className = 'wide';
        grid.appendChild(img);
      }}
    }});
  }}

  sweepKeys.forEach((k, i) => {{
    const btn = document.createElement('button');
    btn.className = 'pill' + (i === 0 ? ' active' : '');
    btn.dataset.key = k;
    btn.textContent = D.sweeps[k].label;
    btn.style.borderColor = D.sweeps[k].color;
    btn.addEventListener('click', () => show(k));
    pills.appendChild(btn);
  }});
  if (sweepKeys.length) show(sweepKeys[0]);
}})();

/* ---- COMPARISON PLOTS ---- */
(function() {{
  const el = document.getElementById('compPlots');
  const plots = D.comparison_plots || {{}};
  ['all_sweeps_comparison_petri','all_sweeps_comparison_mgs'].forEach(name => {{
    if (plots[name]) {{
      const img = document.createElement('img');
      img.src = plots[name];
      img.className = 'comp-img';
      img.alt = name;
      el.appendChild(img);
    }}
  }});
  if (!el.children.length) el.innerHTML = '<p style="color:var(--t3)">No comparison plots found.</p>';
}})();

/* ---- NAV HIGHLIGHT ---- */
(function() {{
  const links = document.querySelectorAll('.sidebar nav a');
  const sections = [...links].map(a => document.querySelector(a.getAttribute('href')));
  const obs = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        links.forEach(a => a.classList.toggle('active',
          a.getAttribute('href') === '#' + e.target.id));
      }}
    }});
  }}, {{ rootMargin: '-30% 0px -60% 0px' }});
  sections.forEach(s => {{ if (s) obs.observe(s); }});
}})();
</script>
</body>
</html>"""


def main():
    print("Loading sweep data...")
    data = load_dashboard_data()
    print(f"\nGenerating dashboard ({len(data['sweeps'])} sweeps, "
          f"{len(data['baselines'])} baselines)...")
    html = generate_html(data)
    out = OUTPUTS / "dashboard.html"
    out.write_text(html)
    print(f"Dashboard saved to {out}")
    print(f"Open with: open {out}")


if __name__ == "__main__":
    main()
