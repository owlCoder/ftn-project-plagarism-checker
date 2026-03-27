#!/usr/bin/env python3
"""
Plagiarism Checker – Reliable Jaccard‑Based Detection
-----------------------------------------------------
- Uses the same simple tokenization and shingling as your original working script
- Overall similarity = Jaccard on the union of all file shingles (global)
- File‑level similarity = Jaccard on per‑file shingles
- Cross‑path files considered if similarity >= cross_thresh (configurable)
- Parallel processing, clean HTML report
"""

import hashlib
import html
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from difflib import HtmlDiff
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import argparse
import multiprocessing as mp
import itertools

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_ALLOWED_EXTS = {'py', 'java', 'js', 'ts', 'cpp', 'c', 'cs', 'sql'}
DEFAULT_SHINGLE_SIZE = 5          # token n‑gram size
DEFAULT_FILE_THRESHOLD = 0.6      # per‑file similarity threshold (same‑path)
DEFAULT_OVERALL_THRESHOLD = 0.4   # overall similarity threshold
DEFAULT_CROSS_THRESHOLD = 0.7     # cross‑path files are considered if sim >= this
DEFAULT_TERMS = ['any', 'exception', 'todo']
MIN_FILE_LINES = 10

EXCLUDED_DIRS = {
    'dist-electron', 'dist', 'public', '__pycache__', '.git', 'node_modules',
    'build', 'target', 'out', 'bin', 'obj', 'venv', 'env', '.idea', '.vscode'
}

# ----------------------------------------------------------------------
# Normalization (simple: strip comments, strings, whitespace)
# ----------------------------------------------------------------------
_STRIP_LINE_COMMENT  = re.compile(r'//.*')
_STRIP_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRIP_HASH_COMMENT  = re.compile(r'#.*')
_STRIP_STRINGS       = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|".*?"|\'.*?\')', re.DOTALL)
_WHITESPACE          = re.compile(r'\s+')

@lru_cache(maxsize=10000)
def normalize_content(content: str, ext: str) -> str:
    """Strip comments, string contents, and extra whitespace."""
    if ext in ('py',):
        content = _STRIP_HASH_COMMENT.sub('', content)
    else:
        content = _STRIP_BLOCK_COMMENT.sub('', content)
        content = _STRIP_LINE_COMMENT.sub('', content)
    content = _STRIP_STRINGS.sub('""', content)
    content = _WHITESPACE.sub(' ', content).strip()
    return content.lower()

def tokenize_normalized(content: str, ext: str) -> List[str]:
    """Split into alphanumeric tokens (identifiers, keywords, numbers)."""
    norm = normalize_content(content, ext)
    return re.findall(r'[a-z0-9_]+', norm)

def shingles(tokens: List[str], k: int = DEFAULT_SHINGLE_SIZE) -> Set[str]:
    """k‑shingle set: every consecutive k‑gram as a single string."""
    if len(tokens) < k:
        return {' '.join(tokens)}
    return {' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def file_similarity(src_a: str, src_b: str, ext: str) -> float:
    """Return Jaccard similarity of two source files, 0–1."""
    ta = shingles(tokenize_normalized(src_a, ext))
    tb = shingles(tokenize_normalized(src_b, ext))
    return jaccard(ta, tb)

# ----------------------------------------------------------------------
# Hashing for Template Matching
# ----------------------------------------------------------------------
@lru_cache(maxsize=10000)
def normalized_hash(content: str, ext: str) -> str:
    norm = normalize_content(content, ext)
    return hashlib.sha256(norm.encode()).hexdigest()

# ----------------------------------------------------------------------
# ZIP Processing
# ----------------------------------------------------------------------
@dataclass
class Student:
    name: str
    all_files: Dict[str, str] = field(default_factory=dict)
    student_files: Dict[str, str] = field(default_factory=dict)
    total_lines: int = 0
    student_lines: int = 0
    template_pct: float = 0.0
    term_counts: Dict[str, int] = field(default_factory=dict)

def should_skip_path(path: str, exclude_dirs: Set[str]) -> bool:
    parts = path.split('/')
    for part in parts:
        if part.lower() in exclude_dirs:
            return True
    return False

def strip_common_prefix(files: Dict[str, str]) -> Dict[str, str]:
    paths = list(files.keys())
    if not paths:
        return files
    parts = [p.split('/') for p in paths]
    if all(p[0] == parts[0][0] for p in parts):
        return { '/'.join(p[1:]): files[path] for path, p in zip(paths, parts) }
    return files

def process_template(zip_path: Optional[Path], allowed_exts: Set[str],
                     exclude_dirs: Set[str]) -> Set[str]:
    if zip_path is None:
        return set()
    hashes = set()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir() or should_skip_path(info.filename, exclude_dirs):
                continue
            ext = info.filename.split('.')[-1].lower()
            if ext not in allowed_exts:
                continue
            with zf.open(info) as f:
                content = f.read().decode('utf-8', errors='ignore')
                hashes.add(normalized_hash(content, ext))
    return hashes

def process_student(zip_path: Path, allowed_exts: Set[str], template_hashes: Set[str],
                    exclude_dirs: Set[str]) -> Student:
    name = zip_path.stem
    all_files = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir() or should_skip_path(info.filename, exclude_dirs):
                continue
            ext = info.filename.split('.')[-1].lower()
            if ext not in allowed_exts:
                continue
            with zf.open(info) as f:
                content = f.read().decode('utf-8', errors='ignore')
                all_files[info.filename] = content

    all_files = strip_common_prefix(all_files)

    student_files = {}
    for path, content in all_files.items():
        ext = path.split('.')[-1].lower()
        if normalized_hash(content, ext) not in template_hashes:
            # Skip very short files
            if content.count('\n') >= MIN_FILE_LINES:
                student_files[path] = content

    total_lines = sum(c.count('\n') for c in all_files.values())
    student_lines = sum(c.count('\n') for c in student_files.values())
    template_pct = 0.0 if total_lines == 0 else 100.0 * (total_lines - student_lines) / total_lines

    return Student(
        name=name,
        all_files=all_files,
        student_files=student_files,
        total_lines=total_lines,
        student_lines=student_lines,
        template_pct=template_pct,
    )

# ----------------------------------------------------------------------
# Term Checking
# ----------------------------------------------------------------------
def check_terms(student: Student, term_list: List[str]) -> Dict[str, int]:
    counts = {t: 0 for t in term_list}
    for content in student.student_files.values():
        lower = content.lower()
        for term in term_list:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            counts[term] += len(pattern.findall(lower))
    return counts

# ----------------------------------------------------------------------
# Pair Comparison (using simple Jaccard)
# ----------------------------------------------------------------------
def compare_pair(stud_a: Student, stud_b: Student, file_thresh: float, cross_thresh: float) -> dict:
    flagged = []
    all_file_sims = []  # collect all similarities for overall score

    # Same‑path files
    common = set(stud_a.student_files.keys()) & set(stud_b.student_files.keys())
    for path in common:
        ext = path.split('.')[-1].lower()
        sim = file_similarity(stud_a.student_files[path], stud_b.student_files[path], ext)
        all_file_sims.append(sim)
        if sim >= file_thresh:
            flagged.append({
                'pathA': path, 'pathB': path,
                'similarity': sim,
                'contentA': stud_a.student_files[path],
                'contentB': stud_b.student_files[path],
            })

    # Cross‑path files (different names) – only if sim >= cross_thresh
    cross = set()
    for pa, ca in stud_a.student_files.items():
        for pb, cb in stud_b.student_files.items():
            if pa == pb:
                continue
            key = f"{pa}|{pb}"
            if key in cross:
                continue
            cross.add(key)
            ext_a = pa.split('.')[-1].lower()
            ext_b = pb.split('.')[-1].lower()
            if ext_a != ext_b:
                continue
            sim = file_similarity(ca, cb, ext_a)
            all_file_sims.append(sim)
            if sim >= cross_thresh:
                flagged.append({
                    'pathA': pa, 'pathB': pb,
                    'similarity': sim,
                    'contentA': ca,
                    'contentB': cb,
                })

    flagged.sort(key=lambda x: x['similarity'], reverse=True)

    # Overall similarity = average of top 5 file similarities (or all if fewer)
    if all_file_sims:
        top = sorted(all_file_sims, reverse=True)[:5]
        overall = sum(top) / len(top)
    else:
        overall = 0.0

    return {
        'studentA': stud_a.name,
        'studentB': stud_b.name,
        'overallSimilarity': overall,
        'flaggedFiles': flagged[:20],
        'totalLinesA': stud_a.student_lines,
        'totalLinesB': stud_b.student_lines,
    }

# ----------------------------------------------------------------------
# Main Analysis (all‑pairs)
# ----------------------------------------------------------------------
def run_analysis(template_path: Optional[Path], student_paths: List[Path],
                 allowed_exts: Set[str], term_list: List[str],
                 file_thresh: float, overall_thresh: float, cross_thresh: float,
                 exclude_dirs: Set[str] = None, num_workers: int = None) -> Tuple[List[Student], List[dict]]:
    if num_workers is None:
        num_workers = mp.cpu_count()
    if exclude_dirs is None:
        exclude_dirs = set()

    # 1. Template
    template_hashes = process_template(template_path, allowed_exts, exclude_dirs)
    print(f"Template hashes: {len(template_hashes)} unique files.")

    # 2. Process students (I/O heavy, use ThreadPool)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_student, p, allowed_exts, template_hashes, exclude_dirs) for p in student_paths]
        students = [f.result() for f in futures]

    # 3. Term counts
    for s in students:
        s.term_counts = check_terms(s, term_list)

    # 4. Compare all student pairs
    pairs = []
    total_pairs = len(students)*(len(students)-1)//2
    print(f"Comparing {total_pairs} pairs using {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(len(students)):
            for j in range(i+1, len(students)):
                futures.append(executor.submit(compare_pair, students[i], students[j], file_thresh, cross_thresh))
        for future in futures:
            pair = future.result()
            if pair['overallSimilarity'] >= overall_thresh:
                pairs.append(pair)

    pairs.sort(key=lambda x: x['overallSimilarity'], reverse=True)
    print(f"Kept {len(pairs)} pairs with overall similarity >= {overall_thresh:.2f}.")
    return students, pairs

# ----------------------------------------------------------------------
# HTML Report Generation (same as before, using HtmlDiff)
# ----------------------------------------------------------------------
def generate_html_report(students: List[Student], pairs: List[dict],
                         term_list: List[str], output_path: Path):
    html_diff = HtmlDiff(tabsize=4, wrapcolumn=80)

    def sim_color(sim):
        pct = sim * 100
        if pct >= 80:
            return '#dc2626'
        elif pct >= 60:
            return '#f97316'
        elif pct >= 40:
            return '#eab308'
        else:
            return '#10b981'

    diff_style = """
        .diff-container {
            margin-top: 0.5rem;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            overflow: hidden;
            background-color: #ffffff;
        }
        .diff-header {
            background-color: #f9fafb;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .diff-header code {
            background: #f3f4f6;
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
        }
        .diff-content {
            overflow-x: auto;
            background: #ffffff;
        }
        .diff-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            font-size: 0.75rem;
            line-height: 1.4;
        }
        .diff-table td {
            padding: 0.25rem 0.5rem;
            vertical-align: top;
            border-bottom: 1px solid #e5e7eb;
        }
        .diff-table .diff_header {
            background-color: #f3f4f6;
            color: #374151;
            font-weight: normal;
            text-align: center;
            padding: 0.25rem;
        }
        .diff-table .diff_add {
            background-color: #d1fae5;
            color: #065f46;
        }
        .diff-table .diff_sub {
            background-color: #fee2e2;
            color: #991b1b;
        }
        .diff-table .diff_chg {
            background-color: #fef3c7;
            color: #92400e;
        }
        .diff-table .diff_ctx {
            background-color: #ffffff;
        }
        .diff-table .diff_next {
            background-color: #f3f4f6;
            color: #6b7280;
            text-align: center;
        }
        .diff-table .diff_lineno {
            text-align: right;
            color: #9ca3af;
            user-select: none;
            width: 45px;
            border-right: 1px solid #e5e7eb;
        }
        .diff-empty {
            padding: 1rem;
            text-align: center;
            color: #6b7280;
            font-style: italic;
        }
    """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Checker Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        {diff_style}
        .similarity-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            color: white;
            font-weight: bold;
            display: inline-block;
            font-size: 0.875rem;
        }}
        .card {{
            transition: all 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        button.diff-toggle {{
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 0.375rem;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: background 0.2s;
        }}
        button.diff-toggle:hover {{
            background: #2563eb;
        }}
        .file-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .file-badge-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">Plagiarism Checker Report</h1>
        <p class="text-gray-600 mb-8">Analysis of {len(students)} student submissions</p>

        <!-- Student Overview -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Student Overview</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
"""
    for s in students:
        html_content += f"""
                <div class="card bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <div class="font-medium text-gray-800">{html.escape(s.name)}</div>
                    <div class="text-sm text-gray-600 mt-2">
                        <div>Total lines: {s.total_lines}</div>
                        <div>Student lines: {s.student_lines}</div>
                        <div>Template: {s.template_pct:.1f}%</div>
                    </div>
                </div>
"""
    html_content += """
            </div>
        </div>
"""

    if term_list:
        html_content += """
        <!-- Term Occurrences -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Term Occurrences</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
"""
        for s in students:
            html_content += f"""
                <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <div class="font-medium text-gray-800">{html.escape(s.name)}</div>
                    <div class="mt-2 flex flex-wrap gap-2">
"""
            for t in term_list:
                html_content += f"""
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-200 text-gray-700">
                            {html.escape(t)}: <span class="font-bold ml-1">{s.term_counts[t]}</span>
                        </span>
"""
            html_content += """
                    </div>
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    if pairs:
        html_content += """
        <!-- Pairwise Similarity -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Suspicious Pairs</h2>
            <div class="space-y-4">
"""
        for idx, pair in enumerate(pairs):
            sim_pct = pair['overallSimilarity'] * 100
            color = sim_color(pair['overallSimilarity'])
            html_content += f"""
                <div class="border border-gray-200 rounded-lg overflow-hidden">
                    <div class="bg-gray-50 p-4 cursor-pointer" onclick="togglePair({idx})">
                        <div class="flex justify-between items-center">
                            <div>
                                <span class="font-medium text-gray-800">{html.escape(pair['studentA'])}</span>
                                <span class="mx-2 text-gray-400">↔</span>
                                <span class="font-medium text-gray-800">{html.escape(pair['studentB'])}</span>
                                <span class="text-sm text-gray-500 ml-4">Student lines: {pair['totalLinesA']} / {pair['totalLinesB']}</span>
                            </div>
                            <div class="flex items-center gap-3">
                                <span class="similarity-badge" style="background: {color};">{sim_pct:.1f}%</span>
                                <span id="chevron{idx}" class="text-gray-500">▼</span>
                            </div>
                        </div>
                    </div>
                    <div id="pair{idx}" style="display: none;" class="border-t border-gray-200 p-4 space-y-4">
"""
            for fi, f in enumerate(pair['flaggedFiles']):
                sim_pct = f['similarity'] * 100
                color = sim_color(f['similarity'])
                diff_id = f"diff_{idx}_{fi}"
                old_lines = f['contentA'].splitlines()
                new_lines = f['contentB'].splitlines()
                diff_table = html_diff.make_table(old_lines, new_lines,
                                                  fromdesc=f"{html.escape(pair['studentA'])} - {html.escape(f['pathA'])}",
                                                  todesc=f"{html.escape(pair['studentB'])} - {html.escape(f['pathB'])}",
                                                  context=True, numlines=3)
                if not diff_table.strip():
                    diff_table = '<div class="diff-empty">No differences found (files are identical).</div>'
                diff_html = f"""
                <div class="diff-container">
                    <div class="diff-header">
                        <code>{html.escape(f['pathA'])} ↔ {html.escape(f['pathB'])}</code>
                        <div class="file-badge-group">
                            <span class="similarity-badge text-xs" style="background: {color};">{sim_pct:.0f}%</span>
                            <button class="diff-toggle" onclick="toggleDiff('{diff_id}')">Show Diff</button>
                        </div>
                    </div>
                    <div id="{diff_id}" class="diff-content" style="display: none;">
                        {diff_table}
                    </div>
                </div>
                """
                html_content += diff_html
            html_content += """
                    </div>
                </div>
"""
        html_content += """
            </div>
        </div>
"""
    else:
        html_content += """
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <p class="text-gray-600">No pairs above the similarity threshold.</p>
        </div>
"""

    html_content += """
    </div>
    <script>
        function togglePair(idx) {
            var div = document.getElementById('pair' + idx);
            var chevron = document.getElementById('chevron' + idx);
            if (div.style.display === 'none') {
                div.style.display = 'block';
                chevron.innerHTML = '▲';
            } else {
                div.style.display = 'none';
                chevron.innerHTML = '▼';
            }
        }
        function toggleDiff(diffId) {
            var diffDiv = document.getElementById(diffId);
            if (diffDiv.style.display === 'none') {
                diffDiv.style.display = 'block';
            } else {
                diffDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# ----------------------------------------------------------------------
# Command Line Interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Plagiarism Checker')
    parser.add_argument('--template', type=Path, help='Template ZIP file (optional)')
    parser.add_argument('students', nargs='+', type=Path, help='Student ZIP files')
    parser.add_argument('--output', type=Path, default=Path('report.html'), help='Output HTML file')
    parser.add_argument('--file-thresh', type=float, default=DEFAULT_FILE_THRESHOLD,
                        help=f'File similarity threshold (default: {DEFAULT_FILE_THRESHOLD})')
    parser.add_argument('--overall-thresh', type=float, default=DEFAULT_OVERALL_THRESHOLD,
                        help=f'Overall similarity threshold (default: {DEFAULT_OVERALL_THRESHOLD})')
    parser.add_argument('--cross-thresh', type=float, default=DEFAULT_CROSS_THRESHOLD,
                        help=f'Cross‑path file similarity threshold (default: {DEFAULT_CROSS_THRESHOLD})')
    parser.add_argument('--exts', nargs='+', default=list(DEFAULT_ALLOWED_EXTS),
                        help=f'Allowed file extensions (default: {DEFAULT_ALLOWED_EXTS})')
    parser.add_argument('--terms', nargs='+', default=DEFAULT_TERMS,
                        help=f'Terms to count (default: {DEFAULT_TERMS})')
    args = parser.parse_args()

    allowed_exts = set(args.exts)
    term_list = args.terms

    students_list, pairs = run_analysis(
        template_path=args.template,
        student_paths=args.students,
        allowed_exts=allowed_exts,
        term_list=term_list,
        file_thresh=args.file_thresh,
        overall_thresh=args.overall_thresh,
        cross_thresh=args.cross_thresh,
        exclude_dirs=EXCLUDED_DIRS,
        num_workers=None
    )
    generate_html_report(students_list, pairs, term_list, args.output)
    print(f"Report generated: {args.output}")

if __name__ == '__main__':
    main()