"""Generate interactive HTML dashboard for browsing sweep eval transcripts.

Usage:
    uv run python scripts/generate_dashboard.py
    cd outputs && python3 -m http.server 8080
    # Open http://localhost:8080/dashboard.html
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sdf_inoculation.plotting.loaders import (
    classify_hackers,
    discover_rl_runs,
    load_hack_rates,
)
from src.sdf_inoculation.plotting.style import (
    MGS_EVALS_DEFAULT,
    MGS_EVAL_LABELS,
    PETRI_DIM_LABELS,
    PETRI_DIMS_OVERRIDE,
)

DIMS = PETRI_DIMS_OVERRIDE
OUTPUTS = Path("outputs")

SWEEP_DEFS = [
    ("neutral", "Neutral", "#4878CF"),
    ("neutral_rh_mentioned", "Neutral (RH mentioned)", "#6495ED"),
    ("hacking_okay", "Hacking okay", "#DD8855"),
    ("please_hack", "Please hack", "#6ACC65"),
    ("sdf_neutral", "SDF + Neutral", "#D65F5F"),
    ("sdf_neutral_rh_mentioned", "SDF + Neutral (RH mentioned)", "#C44E9E"),
    ("sdf_hacking_okay", "SDF + Hacking okay", "#B84040"),
    ("sdf_please_hack", "SDF + Please hack", "#8B6914"),
]


def build_index():
    index = {"sweeps": {}, "dims": DIMS,
             "dim_labels": {d: PETRI_DIM_LABELS.get(d, d) for d in DIMS},
             "mgs_evals": MGS_EVALS_DEFAULT,
             "mgs_labels": {e: MGS_EVAL_LABELS.get(e, e) for e in MGS_EVALS_DEFAULT}}

    for sweep_key, label, color in SWEEP_DEFS:
        sweep_dir = OUTPUTS / "runs" / sweep_key
        if not sweep_dir.exists():
            continue
        runs_dir = sweep_dir / "runs"
        if not runs_dir.exists():
            continue

        # Classify hackers
        rl_runs = discover_rl_runs(sweep_dir)
        completed = {}
        for name, path in rl_runs.items():
            mf = path / "metrics.jsonl"
            if mf.exists() and sum(1 for _ in open(mf)) >= 24:
                completed[name] = path
        hackers, _ = classify_hackers(completed) if completed else (set(), set())

        # Hack rates for sparklines
        hack_rate_means = {}
        for name, path in completed.items():
            rates = load_hack_rates(path)
            if rates:
                hack_rate_means[name] = rates[-1]

        sweep_entry = {"label": label, "color": color, "runs": {}}

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            is_hacker = run_name in hackers
            final_hack = hack_rate_means.get(run_name)
            run_entry = {"is_hacker": is_hacker,
                         "final_hack_rate": round(final_hack, 3) if final_hack is not None else None,
                         "steps": {}}

            evals_dir = run_dir / "evals"
            if not evals_dir.exists():
                continue

            # Petri transcripts
            petri_base = evals_dir / "petri"
            if petri_base.exists():
                for step_dir in sorted(petri_base.iterdir()):
                    if not step_dir.is_dir():
                        continue
                    step_name = step_dir.name
                    transcripts = []
                    for tf in sorted(step_dir.glob("transcript_*.json")):
                        try:
                            data = json.loads(tf.read_text())
                            jo = data.get("metadata", {}).get("judge_output", {})
                            scores = jo.get("scores", {})
                            transcripts.append({
                                "path": str(tf.relative_to(OUTPUTS)),
                                "scores": {d: scores.get(d) for d in DIMS if d in scores},
                                "summary": (jo.get("summary") or "")[:300],
                            })
                        except Exception:
                            transcripts.append({"path": str(tf.relative_to(OUTPUTS)),
                                                "scores": {}, "summary": ""})
                    if transcripts:
                        run_entry["steps"].setdefault(step_name, {})["petri"] = transcripts

            # MGS evals
            mgs_base = evals_dir / "mgs"
            if mgs_base.exists():
                for step_dir in sorted(mgs_base.iterdir()):
                    if not step_dir.is_dir():
                        continue
                    step_name = step_dir.name
                    summary_file = step_dir / "summary.json"
                    mgs_entry = {}
                    if summary_file.exists():
                        try:
                            s = json.loads(summary_file.read_text())
                            mgs_entry["summary"] = s.get("evals", {})
                        except Exception:
                            pass
                    log_files = sorted(step_dir.glob("logs_*/logs.json"))
                    if log_files:
                        mgs_entry["logs_path"] = str(log_files[-1].relative_to(OUTPUTS))
                    if mgs_entry:
                        run_entry["steps"].setdefault(step_name, {})["mgs"] = mgs_entry

            if run_entry["steps"]:
                sweep_entry["runs"][run_name] = run_entry

        if sweep_entry["runs"]:
            index["sweeps"][sweep_key] = sweep_entry
            n_runs = len(sweep_entry["runs"])
            n_hk = sum(1 for r in sweep_entry["runs"].values() if r["is_hacker"])
            print(f"  {label}: {n_runs} runs ({n_hk} hackers)")

    return index


def generate_html(index):
    d = json.dumps(index)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDF Inoculation — Eval Browser</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#080814;--bg2:#0e0e20;--bg3:rgba(18,18,42,.7);--bg4:#14142e;
  --bdr:rgba(255,255,255,.06);--bdr-h:rgba(255,255,255,.13);
  --t1:#e4e4f0;--t2:#9090b0;--t3:#606080;
  --accent:#818cf8;--green:#34d399;--red:#f87171;--amber:#fbbf24;
  --radius:8px;
}}
html{{scroll-behavior:smooth}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--t1);line-height:1.5;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
::-webkit-scrollbar{{width:5px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--bdr-h);border-radius:3px}}

/* Top bar */
.topbar{{display:flex;align-items:center;gap:16px;padding:10px 20px;
  background:var(--bg2);border-bottom:1px solid var(--bdr);flex-shrink:0;z-index:10}}
.topbar h1{{font-size:13px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--accent);white-space:nowrap}}
.topbar .sep{{width:1px;height:20px;background:var(--bdr)}}

/* Selectors */
.sel-group{{display:flex;align-items:center;gap:6px;font-size:12px}}
.sel-group label{{color:var(--t3);font-weight:500;white-space:nowrap}}
select{{background:var(--bg4);color:var(--t1);border:1px solid var(--bdr);border-radius:5px;
  padding:4px 8px;font-size:12px;font-family:inherit;cursor:pointer;max-width:260px}}
select:hover{{border-color:var(--bdr-h)}}
select:focus{{outline:none;border-color:var(--accent)}}

/* Tab pills */
.tabs{{display:flex;gap:4px}}
.tab{{padding:5px 14px;border-radius:16px;font-size:12px;font-weight:500;
  border:1px solid var(--bdr);background:transparent;color:var(--t2);cursor:pointer;transition:all .15s}}
.tab:hover{{border-color:var(--bdr-h);color:var(--t1)}}
.tab.active{{background:rgba(129,140,248,.12);border-color:var(--accent);color:var(--accent)}}

/* Layout */
.main{{display:flex;flex:1;overflow:hidden}}
.sidebar{{width:260px;flex-shrink:0;border-right:1px solid var(--bdr);
  background:var(--bg2);overflow-y:auto;padding:8px 0}}
.content{{flex:1;overflow-y:auto;padding:20px 24px}}

/* Sidebar items */
.sidebar-section{{padding:4px 12px;font-size:10px;font-weight:600;color:var(--t3);
  text-transform:uppercase;letter-spacing:.5px;margin-top:8px}}
.sidebar-item{{display:flex;align-items:center;gap:8px;padding:6px 14px;
  font-size:12px;color:var(--t2);cursor:pointer;transition:all .12s;border-left:2px solid transparent}}
.sidebar-item:hover{{color:var(--t1);background:rgba(255,255,255,.03)}}
.sidebar-item.active{{color:var(--accent);border-left-color:var(--accent);background:rgba(129,140,248,.06)}}
.sidebar-item .badge{{font-size:10px;padding:1px 6px;border-radius:8px;font-weight:600;margin-left:auto}}
.badge-hacker{{background:rgba(248,113,113,.15);color:var(--red)}}
.badge-clean{{background:rgba(52,211,153,.15);color:var(--green)}}
.sidebar-item .hack-rate{{font-size:10px;color:var(--t3);margin-left:auto;font-variant-numeric:tabular-nums}}

/* Step pills in sidebar */
.step-pills{{display:flex;gap:3px;padding:6px 14px;flex-wrap:wrap}}
.step-pill{{padding:3px 8px;border-radius:4px;font-size:10px;font-weight:500;
  border:1px solid var(--bdr);background:transparent;color:var(--t3);cursor:pointer;transition:all .12s}}
.step-pill:hover{{border-color:var(--bdr-h);color:var(--t2)}}
.step-pill.active{{background:rgba(129,140,248,.12);border-color:var(--accent);color:var(--accent)}}
.step-pill.has-petri{{color:var(--t2)}}
.step-pill.has-mgs{{color:var(--t2)}}

/* Scores bar */
.scores-bar{{display:flex;gap:10px;padding:12px 16px;background:var(--bg3);
  border:1px solid var(--bdr);border-radius:var(--radius);margin-bottom:14px;flex-wrap:wrap;align-items:center}}
.score-chip{{display:flex;flex-direction:column;align-items:center;min-width:60px}}
.score-chip .val{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}}
.score-chip .dim{{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.3px;text-align:center}}

/* Summary */
.summary-box{{padding:12px 16px;background:var(--bg3);border:1px solid var(--bdr);
  border-radius:var(--radius);margin-bottom:14px;font-size:13px;color:var(--t2);line-height:1.6}}
.summary-box h4{{font-size:11px;color:var(--t3);text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}}

/* Conversation */
.conversation{{display:flex;flex-direction:column;gap:8px}}
.msg{{padding:10px 14px;border-radius:var(--radius);font-size:13px;line-height:1.6;
  max-width:85%;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word}}
.msg-system{{background:rgba(129,140,248,.08);border:1px solid rgba(129,140,248,.15);
  align-self:center;max-width:95%;font-size:12px;color:var(--t2)}}
.msg-user{{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.15);align-self:flex-start}}
.msg-assistant{{background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.12);align-self:flex-end}}
.msg-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;
  margin-bottom:4px;opacity:.7}}
.msg-system .msg-label{{color:var(--accent)}}
.msg-user .msg-label{{color:#60a5fa}}
.msg-assistant .msg-label{{color:var(--green)}}
.msg-content{{overflow-wrap:break-word}}
.msg-content pre{{background:rgba(0,0,0,.3);padding:8px 10px;border-radius:4px;
  overflow-x:auto;font-size:12px;margin:6px 0}}
.msg-content code{{background:rgba(0,0,0,.2);padding:1px 4px;border-radius:3px;font-size:12px}}

/* MGS */
.mgs-header{{display:flex;gap:8px;margin-bottom:14px;align-items:center;flex-wrap:wrap}}
.mgs-stat{{font-size:13px;color:var(--t2)}}
.mgs-stat strong{{color:var(--t1)}}
.filter-btn{{padding:4px 10px;border-radius:12px;font-size:11px;font-weight:500;
  border:1px solid var(--bdr);background:transparent;color:var(--t2);cursor:pointer;transition:all .12s}}
.filter-btn:hover{{border-color:var(--bdr-h)}}
.filter-btn.active{{background:rgba(129,140,248,.12);border-color:var(--accent);color:var(--accent)}}

.sample{{background:var(--bg3);border:1px solid var(--bdr);border-radius:var(--radius);
  margin-bottom:8px;overflow:hidden;transition:border-color .15s}}
.sample:hover{{border-color:var(--bdr-h)}}
.sample-hdr{{display:flex;align-items:center;gap:8px;padding:8px 14px;cursor:pointer;font-size:12px}}
.sample-hdr .id{{color:var(--t3);font-variant-numeric:tabular-nums;min-width:50px}}
.sample-hdr .classification{{font-size:10px;font-weight:600;padding:2px 8px;border-radius:8px}}
.sample-hdr .classification.MISALIGNED{{background:rgba(248,113,113,.15);color:var(--red)}}
.sample-hdr .classification.ALIGNED{{background:rgba(52,211,153,.15);color:var(--green)}}
.sample-hdr .explanation{{color:var(--t2);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.sample-hdr .chevron{{color:var(--t3);transition:transform .15s;font-size:10px}}
.sample.open .chevron{{transform:rotate(90deg)}}
.sample-body{{display:none;padding:10px 14px;border-top:1px solid var(--bdr);font-size:13px}}
.sample.open .sample-body{{display:block}}
.sample-body h5{{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:.4px;margin:8px 0 4px}}
.sample-body h5:first-child{{margin-top:0}}
.sample-body .answer{{white-space:pre-wrap;line-height:1.6;color:var(--t1);
  background:rgba(0,0,0,.2);padding:10px;border-radius:6px;margin-bottom:8px;max-height:400px;overflow-y:auto}}
.sample-body .analysis{{color:var(--t2);line-height:1.6;white-space:pre-wrap}}

/* Loading / empty */
.loading,.empty{{display:flex;align-items:center;justify-content:center;
  height:200px;color:var(--t3);font-size:14px}}
.loading::after{{content:'';width:16px;height:16px;border:2px solid var(--bdr-h);
  border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;margin-left:8px}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}

/* Eval type tabs */
.eval-tabs{{display:flex;gap:4px;margin-bottom:14px}}

.footer{{font-size:10px;color:var(--t3);padding:8px 20px;border-top:1px solid var(--bdr);flex-shrink:0;text-align:right}}
</style></head><body>

<div class="topbar">
  <h1>SDF Inoculation</h1>
  <div class="sep"></div>
  <div class="sel-group">
    <label>Sweep</label><select id="selSweep"></select>
  </div>
</div>

<div class="main">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="content" id="content">
    <div class="empty">Select a sweep, run, and step to browse transcripts.</div>
  </div>
</div>

<div class="footer">Generated {ts}</div>

<script>
const IX = {d};
const DIM_LABELS = IX.dim_labels;
const MGS_LABELS = IX.mgs_labels;

// State
let curSweep = null, curRun = null, curStep = null, curEvalType = 'petri';
let mgsFilter = 'all';

/* ---- Helpers ---- */
function scoreColor(v) {{
  if (v <= 3) return 'var(--green)';
  if (v <= 6) return 'var(--amber)';
  return 'var(--red)';
}}
function escHtml(s) {{
  const d = document.createElement('div');
  d.textContent = typeof s === 'string' ? s : JSON.stringify(s, null, 2);
  return d.innerHTML;
}}
function truncate(s, n) {{ return s.length > n ? s.slice(0, n) + '…' : s; }}

/* ---- Sweep selector ---- */
const selSweep = document.getElementById('selSweep');
Object.entries(IX.sweeps).forEach(([k, s]) => {{
  const o = document.createElement('option');
  o.value = k; o.textContent = s.label;
  selSweep.appendChild(o);
}});
selSweep.addEventListener('change', () => {{ curSweep = selSweep.value; curRun = null; curStep = null; renderSidebar(); clearContent(); }});

/* ---- Sidebar ---- */
function renderSidebar() {{
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  if (!curSweep || !IX.sweeps[curSweep]) return;
  const sweep = IX.sweeps[curSweep];

  // Sort runs: hackers first, then by name
  const runKeys = Object.keys(sweep.runs).sort((a, b) => {{
    const ha = sweep.runs[a].is_hacker, hb = sweep.runs[b].is_hacker;
    if (ha !== hb) return ha ? -1 : 1;
    return a.localeCompare(b);
  }});

  const hdr = document.createElement('div');
  hdr.className = 'sidebar-section';
  hdr.textContent = `${{runKeys.length}} runs`;
  sb.appendChild(hdr);

  runKeys.forEach(rk => {{
    const run = sweep.runs[rk];
    // Short name: last part after last underscore group
    const shortName = rk.replace(/^.*?_run/, 'run');

    const item = document.createElement('div');
    item.className = 'sidebar-item' + (curRun === rk ? ' active' : '');
    let badge = '';
    if (run.is_hacker) badge = '<span class="badge badge-hacker">H</span>';
    else badge = '<span class="badge badge-clean">OK</span>';
    const hr = run.final_hack_rate !== null ? `<span class="hack-rate">${{(run.final_hack_rate * 100).toFixed(0)}}%</span>` : '';
    item.innerHTML = `<span>${{shortName}}</span>${{hr}}${{badge}}`;
    item.addEventListener('click', () => {{
      curRun = rk;
      curStep = null;
      renderSidebar();
      renderStepPills();
    }});
    sb.appendChild(item);

    // Show step pills when selected
    if (curRun === rk) {{
      const pills = document.createElement('div');
      pills.className = 'step-pills';
      const steps = Object.keys(run.steps).sort((a, b) => {{
        const na = parseInt(a.replace('s','').replace('final','999'));
        const nb = parseInt(b.replace('s','').replace('final','999'));
        return na - nb;
      }});
      steps.forEach(sk => {{
        const pill = document.createElement('button');
        pill.className = 'step-pill' + (curStep === sk ? ' active' : '');
        const stepData = run.steps[sk];
        if (stepData.petri) pill.classList.add('has-petri');
        if (stepData.mgs) pill.classList.add('has-mgs');
        pill.textContent = sk === 'sfinal' ? 'final' : sk.replace('s', '');
        pill.addEventListener('click', () => {{
          curStep = sk;
          renderSidebar();
          renderContent();
        }});
        pills.appendChild(pill);
      }});
      sb.appendChild(pills);
    }}
  }});
}}

/* ---- Content ---- */
function clearContent() {{
  document.getElementById('content').innerHTML = '<div class="empty">Select a run and step to browse transcripts.</div>';
}}

function renderContent() {{
  const el = document.getElementById('content');
  if (!curSweep || !curRun || !curStep) {{ clearContent(); return; }}
  const stepData = IX.sweeps[curSweep]?.runs[curRun]?.steps[curStep];
  if (!stepData) {{ el.innerHTML = '<div class="empty">No eval data for this step.</div>'; return; }}

  const hasPetri = !!stepData.petri?.length;
  const hasMgs = !!stepData.mgs;

  // Auto-select eval type
  if (curEvalType === 'petri' && !hasPetri && hasMgs) curEvalType = 'mgs';
  if (curEvalType === 'mgs' && !hasMgs && hasPetri) curEvalType = 'petri';

  let html = '<div class="eval-tabs">';
  if (hasPetri) html += `<button class="tab ${{curEvalType==='petri'?'active':''}}" onclick="curEvalType='petri';renderContent()">Petri (${{stepData.petri.length}})</button>`;
  if (hasMgs) html += `<button class="tab ${{curEvalType==='mgs'?'active':''}}" onclick="curEvalType='mgs';renderContent()">MGS</button>`;
  html += '</div>';
  html += '<div id="evalContent"></div>';
  el.innerHTML = html;

  if (curEvalType === 'petri' && hasPetri) renderPetri(stepData.petri);
  else if (curEvalType === 'mgs' && hasMgs) renderMgs(stepData.mgs);
}}

/* ---- Petri ---- */
function renderPetri(transcripts) {{
  const el = document.getElementById('evalContent');
  if (transcripts.length === 1) {{
    renderPetriTranscript(transcripts[0]);
  }} else {{
    let html = '';
    transcripts.forEach((t, i) => {{
      const overall = Object.values(t.scores).length
        ? (Object.values(t.scores).reduce((a,b)=>a+b,0) / Object.values(t.scores).length).toFixed(1) : '—';
      html += `<div class="sample" onclick="renderPetriTranscript(IX.sweeps[curSweep].runs[curRun].steps[curStep].petri[${{i}}])">
        <div class="sample-hdr">
          <span class="id">#${{i+1}}</span>
          <span style="font-weight:600;color:${{scoreColor(parseFloat(overall))}}">${{overall}}</span>
          <span class="explanation">${{escHtml(truncate(t.summary, 120))}}</span>
          <span class="chevron">▶</span>
        </div></div>`;
    }});
    el.innerHTML = html;
  }}
}}

async function renderPetriTranscript(info) {{
  const el = document.getElementById('evalContent');
  el.innerHTML = '<div class="loading">Loading transcript</div>';

  try {{
    const resp = await fetch(info.path);
    const data = await resp.json();
    const jo = data.metadata?.judge_output || {{}};
    const scores = jo.scores || {{}};

    // Scores bar
    let html = '<div class="scores-bar">';
    const overall = Object.values(scores).length
      ? Object.values(scores).reduce((a,b)=>a+b,0) / Object.values(scores).length : 0;
    html += `<div class="score-chip"><div class="val" style="color:${{scoreColor(overall)}}">${{overall.toFixed(1)}}</div><div class="dim">Overall</div></div>`;
    html += '<div style="width:1px;height:30px;background:var(--bdr);margin:0 4px"></div>';
    IX.dims.forEach(d => {{
      const v = scores[d];
      if (v !== undefined) html += `<div class="score-chip"><div class="val" style="color:${{scoreColor(v)}}">${{v}}</div><div class="dim">${{DIM_LABELS[d]}}</div></div>`;
    }});
    html += '</div>';

    // Summary
    if (jo.summary) {{
      html += `<div class="summary-box"><h4>Judge Summary</h4>${{escHtml(jo.summary)}}</div>`;
    }}
    if (jo.justification) {{
      html += `<div class="summary-box"><h4>Justification</h4>${{escHtml(jo.justification)}}</div>`;
    }}

    // Extract conversation from events
    const messages = [];
    const seen = new Set();
    for (const event of data.events || []) {{
      if (event.type !== 'transcript_event') continue;
      if (event.view !== 'target') continue;
      for (const patch of event.edit?.patch || []) {{
        if (patch.op === 'add' && patch.value?.role) {{
          const v = patch.value;
          const mid = v.id || '';
          if (mid && seen.has(mid)) continue;
          if (mid) seen.add(mid);
          if (v.content !== undefined && v.content !== null && v.content !== '')
            messages.push({{ role: v.role, content: v.content }});
        }}
      }}
    }}

    html += '<div class="conversation">';
    messages.forEach(m => {{
      const cls = m.role === 'system' ? 'msg-system' : m.role === 'user' ? 'msg-user' : 'msg-assistant';
      const label = m.role === 'system' ? 'System' : m.role === 'user' ? 'Auditor' : 'Target';
      let content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content, null, 2);
      // Handle array content (tool calls)
      if (Array.isArray(m.content)) {{
        content = m.content.map(c => typeof c === 'string' ? c : c.text || JSON.stringify(c)).join('\\n');
      }}
      html += `<div class="msg ${{cls}}"><div class="msg-label">${{label}}</div><div class="msg-content">${{escHtml(content)}}</div></div>`;
    }});
    html += '</div>';

    el.innerHTML = html;
  }} catch (e) {{
    el.innerHTML = `<div class="empty">Failed to load transcript: ${{e.message}}</div>`;
  }}
}}

/* ---- MGS ---- */
async function renderMgs(mgsInfo) {{
  const el = document.getElementById('evalContent');

  // Show summary first
  let headerHtml = '<div class="mgs-header">';
  if (mgsInfo.summary) {{
    IX.mgs_evals.forEach(e => {{
      const ev = mgsInfo.summary[e];
      if (ev) {{
        const pct = (ev.rate * 100).toFixed(0);
        headerHtml += `<div class="mgs-stat">${{MGS_LABELS[e]}}: <strong style="color:${{ev.rate > 0.3 ? 'var(--red)' : 'var(--green)'}}">${{pct}}%</strong> (${{ev.misaligned}}/${{ev.total}})</div>`;
      }}
    }});
  }}
  headerHtml += '</div>';

  if (!mgsInfo.logs_path) {{
    el.innerHTML = headerHtml + '<div class="empty">No detailed logs available.</div>';
    return;
  }}

  el.innerHTML = headerHtml + '<div class="loading">Loading samples</div>';

  try {{
    const resp = await fetch(mgsInfo.logs_path);
    const data = await resp.json();

    // Build eval type tabs and samples
    const evalTypes = [];
    for (const [key, val] of Object.entries(data)) {{
      if (!val.reductions?.length) continue;
      const samples = val.reductions[0].samples || [];
      let evalName = 'unknown';
      if (key.includes('monitor-disruption')) evalName = 'monitor_disruption';
      else if (key.includes('frame-colleague')) evalName = 'frame_colleague';
      evalTypes.push({{ key, evalName, samples }});
    }}

    let html = headerHtml;

    // Eval type tabs
    html += '<div class="eval-tabs">';
    evalTypes.forEach((et, i) => {{
      html += `<button class="tab ${{i===0?'active':''}}" data-mgs-eval="${{i}}" onclick="switchMgsEval(${{i}})">${{MGS_LABELS[et.evalName] || et.evalName}} (${{et.samples.length}})</button>`;
    }});
    html += '</div>';

    // Filter buttons
    html += '<div class="mgs-header" id="mgsFilters"></div>';
    html += '<div id="mgsSamples"></div>';

    el.innerHTML = html;

    // Store data globally for filtering
    window._mgsEvalTypes = evalTypes;
    window._mgsCurrentEval = 0;
    if (evalTypes.length) switchMgsEval(0);
  }} catch (e) {{
    el.innerHTML = headerHtml + `<div class="empty">Failed to load logs: ${{e.message}}</div>`;
  }}
}}

window.switchMgsEval = function(idx) {{
  window._mgsCurrentEval = idx;
  document.querySelectorAll('[data-mgs-eval]').forEach(b => b.classList.toggle('active', +b.dataset.mgsEval === idx));
  mgsFilter = 'all';
  renderMgsSamples();
}};

function renderMgsSamples() {{
  const et = window._mgsEvalTypes?.[window._mgsCurrentEval];
  if (!et) return;
  const samples = et.samples;
  const nMis = samples.filter(s => s.value === 0).length;

  // Filters
  const filtersEl = document.getElementById('mgsFilters');
  filtersEl.innerHTML = `
    <button class="filter-btn ${{mgsFilter==='all'?'active':''}}" onclick="mgsFilter='all';renderMgsSamples()">All (${{samples.length}})</button>
    <button class="filter-btn ${{mgsFilter==='mis'?'active':''}}" onclick="mgsFilter='mis';renderMgsSamples()">Misaligned (${{nMis}})</button>
    <button class="filter-btn ${{mgsFilter==='aln'?'active':''}}" onclick="mgsFilter='aln';renderMgsSamples()">Aligned (${{samples.length - nMis}})</button>`;

  const filtered = samples.filter(s => {{
    if (mgsFilter === 'mis') return s.value === 0;
    if (mgsFilter === 'aln') return s.value !== 0;
    return true;
  }});

  const container = document.getElementById('mgsSamples');
  let html = '';
  filtered.forEach((s, i) => {{
    const cls = s.metadata?.classification || (s.value === 0 ? 'MISALIGNED' : 'ALIGNED');
    html += `<div class="sample" onclick="this.classList.toggle('open')">
      <div class="sample-hdr">
        <span class="id">${{s.sample_id || '#'+i}}</span>
        <span class="classification ${{cls}}">${{cls}}</span>
        <span class="explanation">${{escHtml(truncate(s.explanation || '', 100))}}</span>
        <span class="chevron">▶</span>
      </div>
      <div class="sample-body">
        <h5>Model Response</h5>
        <div class="answer">${{escHtml(s.answer || '')}}</div>
        ${{s.metadata?.reasoning_analysis ? '<h5>Reasoning Analysis</h5><div class="analysis">' + escHtml(s.metadata.reasoning_analysis) + '</div>' : ''}}
        ${{s.metadata?.emails_analysis ? '<h5>Emails Analysis</h5><div class="analysis">' + escHtml(s.metadata.emails_analysis) + '</div>' : ''}}
      </div>
    </div>`;
  }});
  container.innerHTML = html || '<div class="empty">No samples match filter.</div>';
}}

/* ---- Init ---- */
if (Object.keys(IX.sweeps).length) {{
  selSweep.value = Object.keys(IX.sweeps)[0];
  curSweep = selSweep.value;
  renderSidebar();
}}
</script></body></html>"""


def main():
    print("Building index...")
    index = build_index()
    total_petri = sum(
        len(t) for s in index["sweeps"].values()
        for r in s["runs"].values()
        for st in r["steps"].values()
        for t in [st.get("petri", [])]
    )
    total_mgs = sum(
        1 for s in index["sweeps"].values()
        for r in s["runs"].values()
        for st in r["steps"].values()
        if "mgs" in st
    )
    print(f"\nIndex: {len(index['sweeps'])} sweeps, {total_petri} Petri transcripts, {total_mgs} MGS evals")
    print("Generating dashboard...")
    html = generate_html(index)
    out = OUTPUTS / "dashboard.html"
    out.write_text(html)
    size_kb = out.stat().st_size / 1024
    print(f"Dashboard saved to {out} ({size_kb:.0f}KB)")


if __name__ == "__main__":
    main()
