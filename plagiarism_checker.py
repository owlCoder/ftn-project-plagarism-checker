#!/usr/bin/env python3
"""
plagiarism_checker.py
---------------------
Detects plagiarism between student project ZIP files.
Template code is subtracted before comparison so only
student-written code is measured.

Usage:
    python plagiarism_checker.py <submissions_folder> <template_zip> [options]

Options:
    --threshold   Minimum similarity % to flag as suspicious (default: 40)
    --output      Output HTML file path (default: plagiarism_report.html)
    --extensions  Comma-separated file extensions to analyse
                  (default: ts,tsx,js,jsx,py,java,cs,cpp,c,h)

Example:
    python plagiarism_checker.py ./submissions ./project-template.zip --threshold 50
"""

import argparse
import hashlib
import html
import io
import itertools
import os
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── Tokeniser ─────────────────────────────────────────────────────────────────

# Patterns stripped before comparison (comments, string literals, whitespace)
_STRIP_LINE_COMMENT  = re.compile(r'//.*')
_STRIP_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRIP_HASH_COMMENT  = re.compile(r'#.*')
_STRIP_STRINGS       = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', re.DOTALL)
_WHITESPACE          = re.compile(r'\s+')

def normalise(source: str, ext: str) -> str:
    """Strip comments, string contents, and extra whitespace."""
    if ext in ('py',):
        source = _STRIP_HASH_COMMENT.sub('', source)
    else:
        source = _STRIP_BLOCK_COMMENT.sub('', source)
        source = _STRIP_LINE_COMMENT.sub('', source)
    source = _STRIP_STRINGS.sub('""', source)
    source = _WHITESPACE.sub(' ', source).strip()
    return source.lower()


def tokenise(source: str) -> List[str]:
    """Split into alphanumeric tokens (identifiers, keywords, numbers)."""
    return re.findall(r'[a-z0-9_]+', source)


def shingles(tokens: List[str], k: int = 5) -> Set[str]:
    """k-shingle set: every consecutive k-gram as a single string."""
    if len(tokens) < k:
        return {' '.join(tokens)}
    return {' '.join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}


# ── Similarity ────────────────────────────────────────────────────────────────

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def file_similarity(src_a: str, src_b: str, ext: str) -> float:
    """Return Jaccard similarity of two source files, 0–1."""
    ta = shingles(tokenise(normalise(src_a, ext)))
    tb = shingles(tokenise(normalise(src_b, ext)))
    return jaccard(ta, tb)


# ── ZIP extraction ─────────────────────────────────────────────────────────────

def read_zip(zip_path: str, allowed_ext: Set[str]) -> Dict[str, str]:
    """
    Return {relative_path: source_text} for every matching file in the ZIP.
    Strips the top-level folder if one is present (common in GitHub ZIPs).
    """
    files: Dict[str, str] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith('/')]
            # Detect common prefix (GitHub-style root folder)
            parts = [n.split('/') for n in names if '/' in n]
            prefix = parts[0][0] + '/' if parts and all(p[0] == parts[0][0] for p in parts) else ''
            for name in names:
                ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
                if ext not in allowed_ext:
                    continue
                rel = name[len(prefix):] if prefix and name.startswith(prefix) else name
                try:
                    data = zf.read(name)
                    text = data.decode('utf-8', errors='replace')
                    files[rel] = text
                except Exception:
                    pass
    except zipfile.BadZipFile:
        print(f"  [WARN] Cannot open ZIP: {zip_path}", file=sys.stderr)
    return files


# ── Template subtraction ──────────────────────────────────────────────────────

def content_hash(text: str, ext: str) -> str:
    """Normalised content hash used to identify template files."""
    return hashlib.md5(normalise(text, ext).encode()).hexdigest()


def build_template_hashes(template_zip: str, allowed_ext: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    Returns:
        path_set  – relative paths that exist in template
        hash_set  – normalised content hashes of template files
    """
    tpl = read_zip(template_zip, allowed_ext)
    path_set = set(tpl.keys())
    hash_set = {content_hash(v, k.rsplit('.', 1)[-1].lower()) for k, v in tpl.items()}
    return path_set, hash_set


def subtract_template(
    student_files: Dict[str, str],
    tpl_paths: Set[str],
    tpl_hashes: Set[str],
    allowed_ext: Set[str],
) -> Dict[str, str]:
    """
    Remove files that are:
      1. Present by path in the template AND content is unchanged, OR
      2. Identical in normalised content to any template file (renamed copies).
    """
    result = {}
    for path, text in student_files.items():
        ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
        h = content_hash(text, ext)
        if h in tpl_hashes:
            continue          # content unchanged from template
        result[path] = text
    return result


# ── Per-student analysis ──────────────────────────────────────────────────────

@dataclass
class StudentProject:
    name: str                              # student identifier (ZIP filename without .zip)
    all_files: Dict[str, str]             # everything in the ZIP
    student_files: Dict[str, str]         # after template subtraction
    total_lines: int = 0
    student_lines: int = 0

    def __post_init__(self):
        self.total_lines   = sum(len(v.splitlines()) for v in self.all_files.values())
        self.student_lines = sum(len(v.splitlines()) for v in self.student_files.values())

    @property
    def template_pct(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return 100.0 * (self.total_lines - self.student_lines) / self.total_lines


# ── Pairwise comparison ───────────────────────────────────────────────────────

@dataclass
class FilePairMatch:
    path_a: str
    path_b: str
    similarity: float          # 0–1


@dataclass
class PairResult:
    student_a: str
    student_b: str
    overall_similarity: float   # weighted average across shared files
    flagged_files: List[FilePairMatch] = field(default_factory=list)
    total_student_lines_a: int = 0
    total_student_lines_b: int = 0


def compare_pair(
    proj_a: StudentProject,
    proj_b: StudentProject,
    file_threshold: float = 0.6,
) -> PairResult:
    """
    Compare two projects file-by-file (only student-written files).
    Overall similarity = weighted Jaccard over the union of shingles.
    """
    files_a = proj_a.student_files
    files_b = proj_b.student_files

    if not files_a or not files_b:
        return PairResult(proj_a.name, proj_b.name, 0.0,
                          total_student_lines_a=proj_a.student_lines,
                          total_student_lines_b=proj_b.student_lines)

    # Build per-project token pool for overall similarity
    def pool(files: Dict[str, str]) -> Set[str]:
        tokens: List[str] = []
        for path, text in files.items():
            ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
            tokens.extend(tokenise(normalise(text, ext)))
        return shingles(tokens, k=7)

    pool_a = pool(files_a)
    pool_b = pool(files_b)
    overall = jaccard(pool_a, pool_b)

    # Per-file detail (only files present in both or very similar files)
    flagged: List[FilePairMatch] = []
    all_paths = set(files_a) | set(files_b)

    for path in all_paths:
        if path in files_a and path in files_b:
            ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
            sim = file_similarity(files_a[path], files_b[path], ext)
            if sim >= file_threshold:
                flagged.append(FilePairMatch(path, path, sim))

    # Also check cross-path similarity for the highest-value files
    # (catches renamed files)
    if len(files_a) <= 30 and len(files_b) <= 30:
        checked = {(m.path_a, m.path_b) for m in flagged}
        for pa, ta in files_a.items():
            ext_a = pa.rsplit('.', 1)[-1].lower() if '.' in pa else ''
            for pb, tb in files_b.items():
                if pa == pb:
                    continue
                if (pa, pb) in checked:
                    continue
                ext_b = pb.rsplit('.', 1)[-1].lower() if '.' in pb else ''
                if ext_a != ext_b:
                    continue
                sim = file_similarity(ta, tb, ext_a)
                if sim >= 0.85:          # high threshold for cross-path
                    flagged.append(FilePairMatch(pa, pb, sim))
                    checked.add((pa, pb))

    flagged.sort(key=lambda m: m.similarity, reverse=True)

    return PairResult(
        student_a=proj_a.name,
        student_b=proj_b.name,
        overall_similarity=overall,
        flagged_files=flagged[:20],      # cap detail rows
        total_student_lines_a=proj_a.student_lines,
        total_student_lines_b=proj_b.student_lines,
    )


# ── HTML report ───────────────────────────────────────────────────────────────

def pct_colour(pct: float) -> str:
    """Return a CSS color from green → yellow → red based on 0–100 pct."""
    if pct < 30:
        return '#4ade80'   # green
    if pct < 50:
        return '#facc15'   # yellow
    if pct < 70:
        return '#fb923c'   # orange
    return '#f87171'       # red


def sim_colour(sim: float) -> str:
    return pct_colour(sim * 100)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Plagiarism Report</title>
<style>
  :root {{
    --bg: #0f0f0f; --surface: #1a1a1a; --border: #2a2a2a;
    --text: #e5e5e5; --muted: #888; --accent: #6366f1;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
          font-size: 14px; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: .25rem; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin: 2rem 0 .75rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: .08em; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: .5rem .75rem; font-size: .75rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: .06em; border-bottom: 1px solid var(--border); }}
  td {{ padding: .55rem .75rem; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #ffffff08; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
           overflow: hidden; margin-bottom: 1rem; }}
  .pair-header {{ display: flex; align-items: center; justify-content: space-between;
                  padding: .75rem 1rem; cursor: pointer; user-select: none; }}
  .pair-header:hover {{ background: #ffffff06; }}
  .pair-title {{ font-weight: 600; font-size: .95rem; }}
  .pair-meta {{ font-size: .8rem; color: var(--muted); margin-top: .15rem; }}
  .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 6px;
            font-size: .8rem; font-weight: 700; color: #000; }}
  .detail {{ display: none; padding: 0 1rem 1rem; }}
  .detail.open {{ display: block; }}
  .detail table {{ font-size: .82rem; }}
  .bar-wrap {{ width: 120px; height: 7px; background: #2a2a2a; border-radius: 4px; display: inline-block; vertical-align: middle; }}
  .bar {{ height: 7px; border-radius: 4px; }}
  .pill {{ display: inline-block; padding: .15rem .5rem; border-radius: 5px;
           font-size: .72rem; font-weight: 600; margin-right: .3rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: .75rem; margin-bottom: 2rem; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }}
  .stat-label {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; margin-top: .2rem; }}
  .filter-bar {{ margin-bottom: 1rem; display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }}
  input[type=range] {{ accent-color: var(--accent); }}
  #thresh-label {{ font-size: .85rem; color: var(--muted); min-width: 60px; }}
  .no-pairs {{ color: var(--muted); font-size: .9rem; padding: 1rem 0; }}
  code {{ font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: .8rem;
          background: #ffffff0a; padding: .1rem .35rem; border-radius: 4px; }}
  .chevron {{ transition: transform .2s; font-size: .8rem; }}
  .open .chevron {{ transform: rotate(180deg); }}
</style>
</head>
<body>
<h1>📋 Plagiarism Report</h1>
<p class="meta">Generated: {generated} &nbsp;·&nbsp; Template: <code>{template_name}</code>
  &nbsp;·&nbsp; Students: <strong>{n_students}</strong>
  &nbsp;·&nbsp; Pairs analysed: <strong>{n_pairs}</strong></p>

<div class="summary-grid">
  <div class="stat-card">
    <div class="stat-label">Suspicious pairs (&gt;40%)</div>
    <div class="stat-value" style="color:#f87171">{n_suspicious}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg template code %</div>
    <div class="stat-value" style="color:#6366f1">{avg_tpl_pct:.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Max similarity found</div>
    <div class="stat-value" style="color:{max_sim_colour}">{max_sim:.1f}%</div>
  </div>
</div>

<h2>Student overview</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Student</th><th>Total lines</th><th>Student lines</th><th>Template %</th>
  </tr></thead>
  <tbody>
{student_rows}
  </tbody>
</table>
</div>

<h2>Pairwise similarity</h2>
<div class="filter-bar">
  <label style="color:var(--muted);font-size:.85rem">Show pairs above:</label>
  <input type="range" id="thresh" min="0" max="100" value="0" oninput="filterPairs(this.value)"/>
  <span id="thresh-label">0%</span>
</div>
<div id="pairs-container">
{pair_cards}
</div>

<script>
function toggleDetail(id) {{
  const el = document.getElementById(id);
  el.classList.toggle('open');
  const hdr = el.previousElementSibling;
  hdr.classList.toggle('open');
}}
function filterPairs(val) {{
  document.getElementById('thresh-label').textContent = val + '%';
  document.querySelectorAll('.pair-card').forEach(card => {{
    const sim = parseFloat(card.dataset.sim);
    card.style.display = sim >= parseFloat(val) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""

def render_student_rows(projects: List[StudentProject]) -> str:
    rows = []
    for p in sorted(projects, key=lambda x: x.template_pct, reverse=True):
        c = pct_colour(p.template_pct)
        rows.append(
            f'    <tr>'
            f'<td><strong>{html.escape(p.name)}</strong></td>'
            f'<td>{p.total_lines:,}</td>'
            f'<td>{p.student_lines:,}</td>'
            f'<td><span class="badge" style="background:{c}">{p.template_pct:.1f}%</span></td>'
            f'</tr>'
        )
    return '\n'.join(rows)


def render_pair_cards(pairs: List[PairResult]) -> str:
    if not pairs:
        return '<p class="no-pairs">No pairs to display.</p>'
    cards = []
    for i, pr in enumerate(pairs):
        sim_pct = pr.overall_similarity * 100
        c = sim_colour(pr.overall_similarity)
        detail_rows = ''
        for fm in pr.flagged_files:
            fc = sim_colour(fm.similarity)
            bar_w = int(fm.similarity * 100)
            same = '✓' if fm.path_a == fm.path_b else '↔'
            detail_rows += (
                f'<tr>'
                f'<td>{same}</td>'
                f'<td><code>{html.escape(fm.path_a)}</code></td>'
                f'<td><code>{html.escape(fm.path_b)}</code></td>'
                f'<td>'
                f'<span class="bar-wrap"><span class="bar" style="width:{bar_w}%;background:{fc}"></span></span>'
                f' <span style="color:{fc};font-weight:700">{fm.similarity*100:.0f}%</span>'
                f'</td>'
                f'</tr>'
            )
        detail_html = ''
        if detail_rows:
            detail_html = (
                f'<table><thead><tr>'
                f'<th style="width:24px"></th>'
                f'<th>File A</th><th>File B</th><th>Similarity</th>'
                f'</tr></thead><tbody>{detail_rows}</tbody></table>'
            )
        else:
            detail_html = '<p style="color:var(--muted);font-size:.85rem;padding:.5rem 0">No individual file matches above threshold.</p>'

        card = (
            f'<div class="card pair-card" data-sim="{sim_pct:.1f}">'
            f'<div class="pair-header" onclick="toggleDetail(\'d{i}\')">'
            f'  <div>'
            f'    <div class="pair-title">'
            f'      {html.escape(pr.student_a)} &nbsp;↔&nbsp; {html.escape(pr.student_b)}'
            f'    </div>'
            f'    <div class="pair-meta">'
            f'      Student lines: {pr.total_student_lines_a:,} / {pr.total_student_lines_b:,}'
            f'      &nbsp;·&nbsp; Flagged files: {len(pr.flagged_files)}'
            f'    </div>'
            f'  </div>'
            f'  <div style="display:flex;align-items:center;gap:.75rem">'
            f'    <span class="badge" style="background:{c}">{sim_pct:.1f}%</span>'
            f'    <span class="chevron">▼</span>'
            f'  </div>'
            f'</div>'
            f'<div class="detail" id="d{i}">{detail_html}</div>'
            f'</div>'
        )
        cards.append(card)
    return '\n'.join(cards)


def generate_report(
    projects: List[StudentProject],
    pairs: List[PairResult],
    template_name: str,
    output_path: str,
) -> None:
    n_suspicious = sum(1 for p in pairs if p.overall_similarity >= 0.40)
    avg_tpl = (sum(p.template_pct for p in projects) / len(projects)) if projects else 0
    max_sim = max((p.overall_similarity for p in pairs), default=0) * 100

    html_out = _HTML_TEMPLATE.format(
        generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
        template_name=html.escape(template_name),
        n_students=len(projects),
        n_pairs=len(pairs),
        n_suspicious=n_suspicious,
        avg_tpl_pct=avg_tpl,
        max_sim=max_sim,
        max_sim_colour=pct_colour(max_sim),
        student_rows=render_student_rows(projects),
        pair_cards=render_pair_cards(pairs),
    )
    Path(output_path).write_text(html_out, encoding='utf-8')
    print(f"\n✅  Report saved → {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Check plagiarism in student ZIP submissions, subtracting template code.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('submissions', help='Folder containing student ZIP files')
    parser.add_argument('template',    help='Template ZIP file to subtract')
    parser.add_argument('--threshold',  type=float, default=40.0,
                        help='Minimum similarity %% to highlight (default: 40)')
    parser.add_argument('--output',     default='plagiarism_report.html',
                        help='Output HTML file (default: plagiarism_report.html)')
    parser.add_argument('--extensions', default='ts,tsx,js,jsx,py,java,cs,cpp,c,h,sql',
                        help='Comma-separated file extensions to analyse')
    parser.add_argument('--file-threshold', type=float, default=0.6, dest='file_threshold',
                        help='Per-file similarity threshold for flagging (0–1, default: 0.6)')
    args = parser.parse_args()

    allowed_ext = {e.strip().lower() for e in args.extensions.split(',')}
    submissions_dir = Path(args.submissions)

    if not submissions_dir.is_dir():
        sys.exit(f"Error: submissions folder not found: {submissions_dir}")
    if not Path(args.template).is_file():
        sys.exit(f"Error: template ZIP not found: {args.template}")

    # ── Load template ──────────────────────────────────────────
    print(f"📦  Loading template: {args.template}")
    tpl_paths, tpl_hashes = build_template_hashes(args.template, allowed_ext)
    print(f"    {len(tpl_paths)} template files indexed ({len(tpl_hashes)} unique content hashes)")

    # ── Load student ZIPs ──────────────────────────────────────
    zip_files = sorted(submissions_dir.glob('*.zip'))
    if not zip_files:
        sys.exit(f"Error: no ZIP files found in {submissions_dir}")

    print(f"\n👥  Loading {len(zip_files)} student submissions…")
    projects: List[StudentProject] = []
    for zp in zip_files:
        name = zp.stem
        all_files = read_zip(str(zp), allowed_ext)
        student_files = subtract_template(all_files, tpl_paths, tpl_hashes, allowed_ext)
        proj = StudentProject(name=name, all_files=all_files, student_files=student_files)
        print(f"    {name:40s}  total={proj.total_lines:5,} lines  "
              f"student={proj.student_lines:5,} lines  "
              f"template={proj.template_pct:.0f}%")
        projects.append(proj)

    if len(projects) < 2:
        sys.exit("Need at least 2 student ZIPs to compare.")

    # ── Pairwise comparison ────────────────────────────────────
    pairs_total = len(projects) * (len(projects) - 1) // 2
    print(f"\n🔍  Comparing {pairs_total} pairs…")

    pairs: List[PairResult] = []
    for i, (pa, pb) in enumerate(itertools.combinations(projects, 2)):
        result = compare_pair(pa, pb, file_threshold=args.file_threshold)
        pairs.append(result)
        sim_pct = result.overall_similarity * 100
        flag = '🚨' if sim_pct >= args.threshold else '  '
        print(f"    {flag} {pa.name} ↔ {pb.name}: {sim_pct:.1f}%")

    # Sort pairs by similarity descending
    pairs.sort(key=lambda p: p.overall_similarity, reverse=True)

    # ── Summary ────────────────────────────────────────────────
    suspicious = [p for p in pairs if p.overall_similarity * 100 >= args.threshold]
    print(f"\n📊  Summary:")
    print(f"    Pairs analysed  : {len(pairs)}")
    print(f"    Suspicious pairs: {len(suspicious)} (≥{args.threshold:.0f}%)")
    if suspicious:
        print(f"\n    Top suspicious pairs:")
        for p in suspicious[:10]:
            print(f"      {p.student_a} ↔ {p.student_b}: {p.overall_similarity*100:.1f}%")

    # ── Generate report ────────────────────────────────────────
    generate_report(projects, pairs, Path(args.template).name, args.output)


if __name__ == '__main__':
    main()
