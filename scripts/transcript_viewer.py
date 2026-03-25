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

# Featured groups: maps display name -> list of directories (matching final_plot.png bars)
FEATURED = {
    "Base": ["final_base"],
    "SDF": ["final_sdf"],
    "SDF-DPO": ["dpo_step_000050", "dpov3_final"],
    "RL": ["final_rl"],
    "RL-IP": ["final_rl_ip"],
    "SDF-RL": ["final_sdf_rl"],
    "SDF-RL-IP": ["final_sdf_rl_ip"],
    "SDF-Llama-RL": ["llama70b_sdf_llama_rl"],
    "SDF-DPO-RL": [
        "sdf_dpo50_rl_step_000005", "sdf_dpo50_rl2_step_000005",
        "sdf_dpo50_rl_step_000010", "sdf_dpo50_rl2_step_000010",
        "sdf_dpo50_rl_step_000015", "sdf_dpo50_rl2_step_000015",
        "sdf_dpo50_rl_step_000020", "sdf_dpo50_rl2_step_000020",
        "sdf_dpo50_rl_final", "sdf_dpo50_rl2_step_final",
        "dpov2_rl1_000005", "dpov2_rl2_000005",
        "dpov2_rl1_000010", "dpov2_rl2_000010",
        "dpov2_rl1_000015", "dpov2_rl2_000015",
        "dpov2_rl1_000020", "dpov2_rl2_000020",
        "dpov2_rl1_final", "dpov2_rl2_final",
        "dpov3_rl2_000005", "dpov3_rl2_000010",
        "dpov3_rl2_000015", "dpov3_rl2_000020",
        "dpov3_rl2_final",
    ],
}

# Invert: map directory name -> featured group name
FEATURED_DIR_MAP = {}
for name, dirs in FEATURED.items():
    for d in dirs:
        FEATURED_DIR_MAP[d] = name


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
            scores = meta.get("judge_output", {}).get("scores", {})
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


def render_group_rows(group_name, transcripts, collapsed=True):
    """Render HTML rows for a single group."""
    avg_overall = sum(t["overall"] for t in transcripts) / len(transcripts)
    all_dims = set()
    for t in transcripts:
        all_dims.update(t["scores"].keys())

    rows = f"""<tr class="group-header" onclick="this.classList.toggle('collapsed');
        let s=this.nextElementSibling;
        while(s&&!s.classList.contains('group-header')){{s.classList.toggle('hidden');s=s.nextElementSibling}}">
        <td colspan="2"><strong>{html.escape(group_name)}</strong></td>
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


def render_index(root):
    all_groups = load_transcripts(root)

    # Build featured section: merge directories into named groups
    featured_merged = {}
    featured_dirs = set()
    for name, dirs in FEATURED.items():
        merged = []
        for d in dirs:
            if d in all_groups:
                merged.extend(all_groups[d])
                featured_dirs.add(d)
        if merged:
            featured_merged[name] = sorted(merged, key=lambda t: -t["overall"])

    # Featured rows
    featured_rows = ""
    for name in FEATURED:
        if name in featured_merged:
            featured_rows += render_group_rows(name, featured_merged[name])

    # Remaining groups (not in featured)
    other_rows = ""
    for group in sorted(all_groups.keys()):
        if group not in featured_dirs:
            other_rows += render_group_rows(group, all_groups[group])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Petri Transcripts</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 4px; color: #f8fafc; }}
h2 {{ font-size: 15px; font-weight: 500; margin: 24px 0 8px; color: #94a3b8; }}
.subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 8px 12px; color: #94a3b8; border-bottom: 1px solid #334155; font-weight: 500; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
tr:hover {{ background: #1e293b; }}
.group-header {{ cursor: pointer; background: #1e293b; }}
.group-header:hover {{ background: #334155; }}
.group-header td {{ padding: 10px 12px; }}
.hidden {{ display: none; }}
a {{ color: #60a5fa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.score {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #0f172a; font-weight: 600; font-size: 12px; }}
.dim {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #0f172a; font-size: 11px; margin: 1px; }}
.dim-scores {{ max-width: 500px; }}
.seed {{ color: #94a3b8; font-size: 12px; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style></head><body>
<h1>Petri Transcripts</h1>
<div class="subtitle">Click a group to expand transcripts</div>

<h2>Main Results</h2>
<table><thead><tr><th>Model</th><th>Seed</th><th>Overall</th><th>Dimensions</th><th>n</th></tr></thead>
<tbody>{featured_rows}</tbody></table>

<h2>All Transcripts</h2>
<table><thead><tr><th>Directory</th><th>Seed</th><th>Overall</th><th>Dimensions</th><th>n</th></tr></thead>
<tbody>{other_rows}</tbody></table>

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

    messages_html = ""
    all_messages = data.get("messages", [])
    target_messages = data.get("target_messages", [])

    if all_messages:
        for msg in all_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
            source = msg.get("source", "")
            source_label = f' ({source})' if source else ""

            role_class = role.replace(" ", "-")
            messages_html += f"""<div class="message {role_class}">
                <div class="msg-header"><span class="role">{html.escape(role)}{html.escape(source_label)}</span></div>
                <div class="msg-content">{html.escape(str(content))}</div>
            </div>"""

    if target_messages:
        messages_html += '<h2 style="font-size:14px;margin:16px 0 8px;color:#94a3b8">Target Messages</h2>'
        for msg in target_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
            role_class = role.replace(" ", "-")
            messages_html += f"""<div class="message {role_class}">
                <div class="msg-header"><span class="role">{html.escape(role)}</span></div>
                <div class="msg-content">{html.escape(str(content))}</div>
            </div>"""

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
.user {{ background: #1e3a5f; }}
.assistant {{ background: #1e293b; }}
.system {{ background: #2d1b4e; }}
.tool {{ background: #1a2e1a; }}
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
