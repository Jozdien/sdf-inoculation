"""Petri transcript viewer. Run and open in browser (VS Code auto-forwards the port).

Usage: python scripts/transcript_viewer.py [--port 8080] [--dir outputs/petri]
"""

import argparse
import json
import html
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

ROOT = Path("outputs/petri")

_BASE_RL_RUNS = [1, 3, 4, 7, 9, 13, 14, 19, 20, 23, 26, 27, 28, 30, 32, 36, 37, 40, 42, 45, 49, 51]
_SDF_RL_V1_RUNS = [1, 2, 7, 9, 10, 12, 14, 15]
_SDF_RL_V3_RUNS = [1, 2, 4, 7, 9, 10, 11, 12, 14, 15, 16]
_SDF_DPO_RL_V2_RUNS = [10]
_SDF_DPO_RL_V3_RUNS = [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 15, 18, 19, 21, 23, 24]

def _expand(entries):
    d = []
    for prefix, indices in entries:
        if indices is None:
            d.append(prefix)
        else:
            d.extend(f"{prefix}{i}" for i in indices)
    return d

# -- Section definitions --

SECTIONS = [
    {
        "id": "override_highlight",
        "title": "Override Seed — Top 30% Most Misaligned (final_override_combined.png)",
        "subtitle": "Checkpoints that produced the highest override-seed scores. These are the data behind outputs/petri/final_override_combined.png.",
        "groups": {
            "Base Llama": ["combined_v4ba_base_llama"],
            "V15 Base RL": ["combined_v4ba_v15_rl"],
            "Base RL (top 30%)": _expand([("sweep6_sweep_v3_base_run", [28, 26, 19, 27, 1, 40])]),
            "SDF RL (top 30%)": _expand([
                ("sweep6_sweep_sdf_run", [10, 9]),
                ("sweep6_sweep_v3_sdf_run", [7, 9, 2]),
            ]),
            "SDF-DPO RL (top 30%)": _expand([
                ("sweep6_sweep_sdf_dpo_v2_run", [10]),
                ("sweep6_sweep_v3_sdf_dpo_run", [7, 24, 3, 12]),
            ]),
        },
        "collapsed": False,
        "highlight": True,
    },
    {
        "id": "combined_reps",
        "title": "Combined 6-Seed Representatives",
        "subtitle": "Single representative checkpoint per condition, evaluated on all 6 seeds.",
        "groups": {
            "Base Llama": ["combined_v4ba_base_llama"],
            "V15 Base RL": ["combined_v4ba_v15_rl"],
            "V3 Base RL": ["combined_v4ba_base_rl"],
            "SDF RL": ["combined_v4ba_sdf_rl"],
            "SDF-DPO RL": ["combined_v4ba_sdf_dpo_rl"],
        },
        "collapsed": False,
    },
    {
        "id": "deep_eval",
        "title": "Deep Evaluations (75 transcripts each)",
        "subtitle": "In-depth 6-seed evaluations of selected checkpoints.",
        "groups": {
            "Base Llama": ["deep_base_llama"],
            "V15 Base RL": ["deep_v15_rl"],
            "Base RL": _expand([("deep_base_rl_run", [3, 7, 27, 28, 49])]),
            "SDF RL": _expand([
                ("deep_sdf_rl_run", [1, 10, 12, 15]),
                ("deep_sdf_rl_v3run", [7]),
            ]),
            "SDF-DPO RL": _expand([
                ("deep_sdf_dpo_rl_v2run", [10]),
                ("deep_sdf_dpo_rl_v3run", [3, 14, 15, 24]),
            ]),
        },
        "collapsed": False,
    },
    {
        "id": "sweep6",
        "title": "6-Seed Sweep (1 per seed per checkpoint)",
        "subtitle": "Quick 6-seed probe across all RL checkpoints.",
        "groups": {
            "Base RL": _expand([("sweep6_sweep_v3_base_run", _BASE_RL_RUNS)]),
            "SDF RL": _expand([
                ("sweep6_sweep_sdf_run", _SDF_RL_V1_RUNS),
                ("sweep6_sweep_v3_sdf_run", _SDF_RL_V3_RUNS),
            ]),
            "SDF-DPO RL": _expand([
                ("sweep6_sweep_sdf_dpo_v2_run", _SDF_DPO_RL_V2_RUNS),
                ("sweep6_sweep_v3_sdf_dpo_run", _SDF_DPO_RL_V3_RUNS),
            ]),
        },
        "collapsed": True,
    },
    {
        "id": "fulldeep",
        "title": "Full Deep Sweep (many per seed per checkpoint)",
        "subtitle": "Deep evaluation across all RL checkpoints.",
        "groups": {
            "Base RL": _expand([("fulldeep_sweep_v3_base_run", _BASE_RL_RUNS)]),
            "SDF RL": _expand([
                ("fulldeep_sweep_sdf_run", _SDF_RL_V1_RUNS),
                ("fulldeep_sweep_v3_sdf_run", _SDF_RL_V3_RUNS),
            ]),
            "SDF-DPO RL": _expand([
                ("fulldeep_sweep_sdf_dpo_v2_run", _SDF_DPO_RL_V2_RUNS),
                ("fulldeep_sweep_v3_sdf_dpo_run", _SDF_DPO_RL_V3_RUNS),
            ]),
        },
        "collapsed": True,
    },
    {
        "id": "v3_sweep",
        "title": "V3 Sweep (4 seeds, legacy)",
        "subtitle": "Original 4-seed sweep results.",
        "groups": {
            "Base Llama (no RL)": ["sweep_base_llama"],
            "Base RL": _expand([("sweep_v3_base_run", _BASE_RL_RUNS)]),
            "SDF RL": _expand([
                ("sweep_sdf_run", _SDF_RL_V1_RUNS),
                ("sweep_v3_sdf_run", _SDF_RL_V3_RUNS),
            ]),
            "SDF-DPO RL": _expand([
                ("sweep_sdf_dpo_v2_run", _SDF_DPO_RL_V2_RUNS),
                ("sweep_v3_sdf_dpo_run", _SDF_DPO_RL_V3_RUNS),
            ]),
        },
        "collapsed": True,
    },
]

# Legacy groups
_SEEDS = ["cand_monitoring_off_{}", "cand_dev_mode_gradual_{}", "cand_agentic_roleplay_{}"]
_SEEDS_WITH_OVERRIDE = ["v15_final_{}"] + _SEEDS
LEGACY = {
    "Base": [s.format("base") for s in _SEEDS_WITH_OVERRIDE],
    "SDF": [s.format("sdf") for s in _SEEDS_WITH_OVERRIDE],
    "SDF-DPO": [s.format("sdf_dpo") for s in _SEEDS_WITH_OVERRIDE],
    "RL": [s.format("rl") for s in _SEEDS_WITH_OVERRIDE],
    "SDF-RL": [s.format("sdf_rl") for s in _SEEDS_WITH_OVERRIDE],
    "SDF-DPO-RL": ["dpov2_rl1_final_v10dims", "cand_agentic_roleplay_sdf_dpo_rl"],
}

# Build set of all featured dirs and map
FEATURED_DIR_MAP = {}
for section in SECTIONS:
    for name, dirs in section["groups"].items():
        for d in dirs:
            FEATURED_DIR_MAP[d] = name
for name, dirs in LEGACY.items():
    for d in dirs:
        FEATURED_DIR_MAP[d] = f"[legacy] {name}"

# Top-30% override dirs for highlight styling
OVERRIDE_TOP30_DIRS = set()
for section in SECTIONS:
    if section.get("highlight"):
        for dirs in section["groups"].values():
            OVERRIDE_TOP30_DIRS.update(dirs)

# MGS config
MGS_DIR = Path("outputs/mgs")
MGS_CONDITIONS = {
    "Base Llama": ["base_llama"],
    "SDF": ["sdf"],
    "SDF-DPO": ["sdf_dpo"],
    "Base RL": [f"sweep_v3_base_run{i}" for i in _BASE_RL_RUNS],
    "SDF RL": (
        [f"sweep_sdf_run{i}" for i in _SDF_RL_V1_RUNS]
        + [f"sweep_v3_sdf_run{i}" for i in _SDF_RL_V3_RUNS]
    ),
    "SDF-DPO RL": (
        [f"sweep_sdf_dpo_v2_run{i}" for i in _SDF_DPO_RL_V2_RUNS]
        + [f"sweep_v3_sdf_dpo_run{i}" for i in _SDF_DPO_RL_V3_RUNS]
    ),
}
MGS_EVAL_LABELS = {
    "goals": "Goals", "betley": "Betley", "alignment_questions": "Align Q",
    "monitor_disruption": "Monitor", "exfil_offer": "Exfil", "frame_colleague": "Frame",
}
MGS_COLORS = {
    "Base Llama": "#4ECDC4", "SDF": "#96CEB4", "SDF-DPO": "#FFEAA7",
    "Base RL": "#45B7D1", "SDF RL": "#FFA07A", "SDF-DPO RL": "#FF6B6B",
}


def score_color(score):
    if score <= 3: return "#22c55e"
    if score <= 5: return "#eab308"
    if score <= 7: return "#f97316"
    return "#ef4444"


def load_transcripts(root):
    """Load all transcripts, return dict of rel_dir -> list of transcript info."""
    groups = {}
    for dirpath, _, files in os.walk(root):
        jsons = sorted(f for f in files if f.endswith(".json"))
        if not jsons:
            continue
        rel = os.path.relpath(dirpath, root)
        transcripts = []
        for f in jsons:
            with open(os.path.join(dirpath, f)) as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    continue
            meta = data.get("metadata", {})
            scores = (meta.get("judge_output") or {}).get("scores", {})
            if not scores:
                continue
            overall = sum(scores.values()) / len(scores) if scores else 0
            transcripts.append({
                "file": f,
                "path": os.path.join(rel, f),
                "seed": (meta.get("seed_instruction") or "")[:80],
                "target": (meta.get("target_model") or "").split("/")[-1][:40],
                "scores": scores,
                "overall": overall,
            })
        if transcripts:
            groups[rel] = sorted(transcripts, key=lambda t: -t["overall"])
    return groups


def render_group_rows(group_name, transcripts, collapsed=True, highlight=False):
    """Render HTML rows for a single group."""
    avg_overall = sum(t["overall"] for t in transcripts) / len(transcripts)
    all_dims = set()
    for t in transcripts:
        all_dims.update(t["scores"].keys())

    hl_class = " highlight" if highlight else ""
    rows = f"""<tr class="group-header{hl_class}" onclick="this.classList.toggle('collapsed');
        let s=this.nextElementSibling;
        while(s&&!s.classList.contains('group-header')){{s.classList.toggle('hidden');s=s.nextElementSibling}}">
        <td colspan="2"><strong>{'&#9733; ' if highlight else ''}{html.escape(group_name)}</strong></td>
        <td><span class="score" style="background:{score_color(avg_overall)}">{avg_overall:.1f}</span></td>
        <td class="dim-scores">"""

    for dim in sorted(all_dims):
        vals = [t["scores"][dim] for t in transcripts if dim in t["scores"]]
        avg = sum(vals) / len(vals) if vals else 0
        rows += f'<span class="dim" style="background:{score_color(avg)}">{dim.replace("_"," ")[:12]} {avg:.1f}</span> '
    rows += f"</td><td>{len(transcripts)}</td></tr>\n"

    hide_class = "hidden" if collapsed else ""
    for t in transcripts:
        score_pills = " ".join(
            f'<span class="dim" style="background:{score_color(v)}">{k.replace("_"," ")[:12]} {v}</span>'
            for k, v in sorted(t["scores"].items())
        )
        rows += f"""<tr class="transcript-row {hide_class}">
            <td><a href="/transcript/{t['path']}">{t['file'][:30]}</a></td>
            <td class="seed">{html.escape(t['seed'])}</td>
            <td><span class="score" style="background:{score_color(t['overall'])}">{t['overall']:.1f}</span></td>
            <td class="dim-scores">{score_pills}</td>
            <td>1</td></tr>\n"""
    return rows


def render_section(section, all_groups):
    """Render a section of featured groups. Returns (html, set of consumed dirs)."""
    consumed = set()
    merged = {}
    for name, dirs in section["groups"].items():
        m = []
        for d in dirs:
            if d in all_groups:
                m.extend(all_groups[d])
                consumed.add(d)
        if m:
            merged[name] = sorted(m, key=lambda t: -t["overall"])

    if not merged:
        return "", consumed

    is_hl = section.get("highlight", False)
    collapsed = section.get("collapsed", True)
    rows = "".join(
        render_group_rows(n, merged[n], collapsed=collapsed, highlight=is_hl)
        for n in section["groups"] if n in merged
    )

    hl_border = ' style="border-left:3px solid #f59e0b"' if is_hl else ""
    subtitle = f'<div class="subtitle">{html.escape(section.get("subtitle", ""))}</div>' if section.get("subtitle") else ""
    sid = html.escape(section["id"])
    return f"""<div class="section" id="{sid}"{hl_border}>
<h2>{html.escape(section["title"])}</h2>{subtitle}
<table><thead><tr><th>Condition</th><th>Seed</th><th>Overall</th><th>Dimensions</th><th>n</th></tr></thead>
<tbody>{rows}</tbody></table></div>\n""", consumed


def mgs_rate_color(rate):
    if rate <= 0.1:
        return "#22c55e"
    if rate <= 0.3:
        return "#eab308"
    if rate <= 0.5:
        return "#f97316"
    return "#ef4444"


def load_mgs_results():
    results = {}
    for f in MGS_DIR.glob("*/summary.json"):
        try:
            data = json.load(open(f))
            results[data["model_name"]] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def render_mgs_section():
    results = load_mgs_results()
    if not results:
        return ""

    eval_names = list(MGS_EVAL_LABELS.keys())
    header_cells = "".join(f"<th>{MGS_EVAL_LABELS[e]}</th>" for e in eval_names)

    rows = ""
    for cond_name, model_names in MGS_CONDITIONS.items():
        cond_results = [results[m] for m in model_names if m in results]
        if not cond_results:
            continue

        # Aggregate MGS
        mgs_vals = [r["mgs"]["value"] for r in cond_results]
        mgs_mean = sum(mgs_vals) / len(mgs_vals)
        n = len(cond_results)
        color = MGS_COLORS.get(cond_name, "#94a3b8")

        # Per-eval averages
        eval_cells = ""
        for e in eval_names:
            rates = [r["evals"][e]["rate"] for r in cond_results if e in r.get("evals", {})]
            if rates:
                avg = sum(rates) / len(rates)
                eval_cells += f'<td><span class="dim" style="background:{mgs_rate_color(avg)}">{avg:.1%}</span></td>'
            else:
                eval_cells += "<td>-</td>"

        rows += f"""<tr class="group-header" style="background:#1e293b" onclick="this.classList.toggle('collapsed');
            let s=this.nextElementSibling;
            while(s&&!s.classList.contains('group-header')){{s.classList.toggle('hidden');s=s.nextElementSibling}}">
            <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px"></span><strong>{html.escape(cond_name)}</strong></td>
            <td><span class="score" style="background:{mgs_rate_color(mgs_mean)}">{mgs_mean:.1%}</span></td>
            {eval_cells}
            <td>{n}</td></tr>\n"""

        # Individual model rows (collapsed) for multi-checkpoint conditions
        if n > 1:
            for r in sorted(cond_results, key=lambda x: -x["mgs"]["value"]):
                ecells = ""
                for e in eval_names:
                    if e in r.get("evals", {}):
                        v = r["evals"][e]["rate"]
                        ecells += f'<td><span class="dim" style="background:{mgs_rate_color(v)}">{v:.1%}</span></td>'
                    else:
                        ecells += "<td>-</td>"
                mval = r["mgs"]["value"]
                rows += f"""<tr class="transcript-row hidden">
                    <td style="padding-left:28px">{html.escape(r["model_name"])}</td>
                    <td><span class="score" style="background:{mgs_rate_color(mval)}">{mval:.1%}</span></td>
                    {ecells}
                    <td>1</td></tr>\n"""

    return f"""<div class="section" id="mgs">
<h2>MGS (Misalignment Generalization Score)</h2>
<div class="subtitle">Aggregate misalignment rate across 6 behavioral evals (~1490 samples per model). Click condition rows to expand.</div>
<table><thead><tr><th>Condition</th><th>MGS</th>{header_cells}<th>n</th></tr></thead>
<tbody>{rows}</tbody></table></div>\n"""


def render_index(root):
    all_groups = load_transcripts(root)
    all_consumed = set()

    # Render each section
    section_html = ""
    banner_section = None
    for section in SECTIONS:
        s_html, consumed = render_section(section, all_groups)
        section_html += s_html
        all_consumed.update(consumed)
        if section["id"] == "combined_reps":
            banner_section = section

    # Legacy
    legacy_consumed = set()
    legacy_merged = {}
    for name, dirs in LEGACY.items():
        m = []
        for d in dirs:
            if d in all_groups:
                m.extend(all_groups[d])
                legacy_consumed.add(d)
        if m:
            legacy_merged[f"[legacy] {name}"] = sorted(m, key=lambda t: -t["overall"])
    all_consumed.update(legacy_consumed)

    legacy_rows = "".join(render_group_rows(n, legacy_merged[n]) for n in legacy_merged)

    # Remaining uncategorized
    other_rows = ""
    for group in sorted(all_groups.keys()):
        if group not in all_consumed:
            other_rows += render_group_rows(group, all_groups[group])

    # Banner from combined reps
    banner_items = []
    if banner_section:
        for name, dirs in banner_section["groups"].items():
            transcripts = []
            for d in dirs:
                if d in all_groups:
                    transcripts.extend(all_groups[d])
            if transcripts:
                avg = sum(t["overall"] for t in transcripts) / len(transcripts)
                n = len(transcripts)
                banner_items.append(f'<div class="stat"><div class="stat-val" style="color:{score_color(avg)}">{avg:.1f}</div><div class="stat-label">{html.escape(name)}<br>({n} transcripts)</div></div>')
    banner = "\n".join(banner_items)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Petri Transcripts</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; color: #f8fafc; }}
h2 {{ font-size: 15px; font-weight: 500; margin: 24px 0 8px; color: #94a3b8; }}
.subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 8px; }}
.section {{ margin-bottom: 16px; padding-left: 8px; }}
.banner {{ display: flex; gap: 24px; background: #1e293b; padding: 20px 24px; border-radius: 12px; margin: 16px 0; flex-wrap: wrap; }}
.stat {{ text-align: center; flex: 1; min-width: 120px; }}
.stat-val {{ font-size: 32px; font-weight: 700; }}
.stat-label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 8px 12px; color: #94a3b8; border-bottom: 1px solid #334155; font-weight: 500; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
tr:hover {{ background: #1e293b; }}
.group-header {{ cursor: pointer; background: #1e293b; }}
.group-header:hover {{ background: #334155; }}
.group-header.highlight {{ background: #2d2006; border-left: 3px solid #f59e0b; }}
.group-header.highlight:hover {{ background: #3d2a08; }}
.group-header td {{ padding: 10px 12px; }}
.hidden {{ display: none; }}
a {{ color: #60a5fa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.score {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #0f172a; font-weight: 600; font-size: 12px; }}
.dim {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #0f172a; font-size: 11px; margin: 1px; }}
.dim-scores {{ max-width: 500px; }}
.seed {{ color: #94a3b8; font-size: 12px; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
nav {{ margin: 12px 0 20px; display: flex; gap: 8px; flex-wrap: wrap; }}
nav a {{ padding: 4px 12px; background: #1e293b; border-radius: 6px; font-size: 12px; }}
nav a:hover {{ background: #334155; }}
</style></head><body>
<h1>Petri Alignment Audit Results</h1>
<div class="subtitle">6 seeds, 6 dimensions. SDF/DPO models score higher under RL reward hacking.</div>

<nav>{"".join(f'<a href="#{s["id"]}">{html.escape(s["title"].split("(")[0].strip())}</a>' for s in SECTIONS)}
<a href="#mgs">MGS</a><a href="#legacy">Legacy</a><a href="#all">All Transcripts</a></nav>

<div class="banner">{banner}</div>

{section_html}

{render_mgs_section()}

<div class="section" id="legacy">
<h2>Legacy Results</h2>
<table><thead><tr><th>Model</th><th>Seed</th><th>Overall</th><th>Dimensions</th><th>n</th></tr></thead>
<tbody>{legacy_rows}</tbody></table></div>

<div class="section" id="all">
<h2>All Transcripts</h2>
<table><thead><tr><th>Directory</th><th>Seed</th><th>Overall</th><th>Dimensions</th><th>n</th></tr></thead>
<tbody>{other_rows}</tbody></table></div>

</body></html>"""


def render_transcript(root, path):
    fpath = root / path
    if not fpath.exists():
        return "<h1>Not found</h1>"

    with open(fpath) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    scores = meta.get("judge_output", {}).get("scores", {})
    summary = meta.get("judge_output", {}).get("summary", "")
    highlights = meta.get("judge_output", {}).get("highlights", [])

    score_pills = " ".join(
        f'<span class="dim" style="background:{score_color(v)}">{k.replace("_"," ")} {v}</span>'
        for k, v in sorted(scores.items())
    )

    meta_html = f"""<div class="meta">
        <div><strong>Target:</strong> {html.escape(str(meta.get('target_model', ''))[:80])}</div>
        <div><strong>Auditor:</strong> {html.escape(str(meta.get('auditor_model', '')))}</div>
        <div><strong>Seed:</strong> {html.escape(str(meta.get('seed_instruction', ''))[:200])}</div>
        <div style="margin-top:8px">{score_pills}</div>
    </div>"""

    if summary:
        meta_html += f'<div class="summary"><strong>Summary:</strong> {html.escape(summary)}</div>'

    if highlights:
        meta_html += '<div class="highlights"><strong>Highlights:</strong><ul>'
        for h in highlights:
            desc = h.get("description", "") if isinstance(h, dict) else str(h)
            meta_html += f"<li>{html.escape(desc)}</li>"
        meta_html += "</ul></div>"

    # Extract messages from target view (auditor↔target conversation)
    # and auditor view (auditor's internal reasoning/tool use)
    def extract_view_messages(view_name):
        msgs = []
        seen = set()
        for event in data.get("events", []):
            if event.get("type") != "transcript_event": continue
            if event.get("view") != view_name: continue
            for patch in event.get("edit", {}).get("patch", []):
                if patch.get("op") != "add": continue
                msg = patch.get("value", {})
                if not isinstance(msg, dict) or "role" not in msg: continue
                msg_id = msg.get("id", "")
                if msg_id in seen: continue
                seen.add(msg_id)
                msgs.append(msg)
        return msgs

    target_msgs = extract_view_messages("target")
    auditor_msgs = extract_view_messages("auditor")

    # If no events, fall back to messages list
    if not target_msgs and not auditor_msgs:
        target_msgs = data.get("target_messages", [])
        auditor_msgs = data.get("messages", [])

    messages_html = ""

    # Render target conversation (the main interaction)
    if target_msgs:
        messages_html += '<h2 style="font-size:14px;margin:16px 0 8px;color:#94a3b8">Target Conversation</h2>'
        for msg in target_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
            # In target view: user = auditor's message TO target, assistant = target's response
            if role == "user":
                css_class = "auditor-msg"
                label = "Auditor"
            elif role == "assistant":
                css_class = "target-msg"
                label = "Target"
            elif role == "system":
                css_class = "system"
                label = "System"
            else:
                css_class = "tool"
                label = role

            messages_html += f"""<div class="message {css_class}">
                <div class="msg-header"><span class="role">{html.escape(label)}</span></div>
                <div class="msg-content">{html.escape(str(content))}</div>
            </div>"""

    # Render auditor reasoning (collapsible)
    if auditor_msgs:
        messages_html += """<details style="margin-top:16px">
            <summary style="cursor:pointer;color:#94a3b8;font-size:14px;font-weight:500;margin-bottom:8px">
            Auditor Internal Reasoning</summary>"""
        for msg in auditor_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
            source = msg.get("source", "")
            source_label = f" ({source})" if source else ""
            role_class = role.replace(" ", "-")
            messages_html += f"""<div class="message {role_class}">
                <div class="msg-header"><span class="role">{html.escape(role)}{html.escape(source_label)}</span></div>
                <div class="msg-content">{html.escape(str(content))}</div>
            </div>"""
        messages_html += "</details>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(path)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; max-width: 900px; margin: 0 auto; }}
a {{ color: #60a5fa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h1 {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #f8fafc; }}
.meta, .summary, .highlights {{ background: #1e293b; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; font-size: 13px; line-height: 1.6; }}
.dim {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #0f172a; font-size: 12px; font-weight: 600; margin: 2px; }}
.message {{ margin-bottom: 8px; padding: 12px 16px; border-radius: 8px; font-size: 13px; line-height: 1.5; }}
.msg-header {{ margin-bottom: 6px; }}
.role {{ font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
.views {{ font-size: 11px; color: #64748b; margin-left: 8px; }}
.msg-content {{ white-space: pre-wrap; word-break: break-word; }}
.auditor-msg {{ background: #1e3a5f; margin-right: 80px; border-left: 3px solid #60a5fa; }}
.target-msg {{ background: #1a2e1a; margin-left: 80px; border-left: 3px solid #34d399; }}
.system {{ background: #2d1b4e; }}
.tool {{ background: #1e293b; }}
.user {{ background: #1e3a5f; }}
.assistant {{ background: #1e293b; }}
.auditor-msg .role {{ color: #60a5fa; }}
.target-msg .role {{ color: #34d399; }}
.user .role {{ color: #60a5fa; }}
.assistant .role {{ color: #34d399; }}
.system .role {{ color: #c084fc; }}
.tool .role {{ color: #86efac; }}
</style></head><body>
<a href="/">&larr; Back</a>
<h1>{html.escape(path)}</h1>
{meta_html}
<h2 style="font-size:14px;margin:16px 0 8px;color:#94a3b8">Conversation</h2>
{messages_html}
</body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path)
        if path == "/" or path == "":
            content = render_index(ROOT)
        elif path.startswith("/transcript/"):
            transcript_path = path[len("/transcript/"):]
            content = render_transcript(ROOT, transcript_path)
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def log_message(self, format, *args):
        pass


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dir", default="outputs/petri")
    args = parser.parse_args()
    ROOT = Path(args.dir)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Transcript viewer: http://localhost:{args.port}")
    print(f"Serving transcripts from: {ROOT}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
