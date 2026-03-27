#!/usr/bin/env python3
"""
Plagiarism Checker – Optimized Jaccard‑Based Detection
-----------------------------------------------------------------------------
- Uses inverted index for cross‑path file comparisons (O(shared shingles))
- Multiprocessing for pair analysis
- Configurable thresholds, shingle size, and exclusions
- HTML report with collapsible diffs (Consolas, 9pt)
"""

import hashlib
import html
import re
import zipfile
import argparse
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from difflib import HtmlDiff
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Callable, Any

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ----------------------------------------------------------------------
# Configuration defaults
# ----------------------------------------------------------------------
DEFAULT_ALLOWED_EXTS = {'py', 'java', 'js', 'ts', 'cpp', 'c', 'cs', 'sql'}
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_FILE_THRESHOLD = 0.6
DEFAULT_OVERALL_THRESHOLD = 0.4
DEFAULT_CROSS_THRESHOLD = 0.7
DEFAULT_TERMS = ['any', 'exception', 'todo']
MIN_FILE_LINES = 10
MAX_FLAGGED_FILES = 20
EXCLUDED_DIRS = {
    'dist-electron', 'dist', 'public', '__pycache__', '.git', 'node_modules',
    'build', 'target', 'out', 'bin', 'obj', 'venv', 'env', '.idea', '.vscode'
}

# ----------------------------------------------------------------------
# Normalization (improved regex)
# ----------------------------------------------------------------------
# Python triple quotes (both """ and ''')
_STRIP_PY_STRINGS = re.compile(
    r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|".*?"|\'.*?\')', re.DOTALL
)
# Multi‑line C‑style comments
_STRIP_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRIP_LINE_COMMENT = re.compile(r'//.*')
_STRIP_HASH_COMMENT = re.compile(r'#.*')
_WHITESPACE = re.compile(r'\s+')

# For languages that use /* ... */ and //, we'll strip in order
def normalize_content(content: str, ext: str) -> str:
    """Strip comments, string contents, and extra whitespace."""
    # 1. Strip strings first (to avoid interfering with comment detection)
    if ext in ('py',):
        content = _STRIP_PY_STRINGS.sub('""', content)
        content = _STRIP_HASH_COMMENT.sub('', content)
    else:
        # C‑style languages: block comments, then line comments
        content = _STRIP_BLOCK_COMMENT.sub('', content)
        content = _STRIP_LINE_COMMENT.sub('', content)
        # Also strip strings (basic)
        content = re.sub(r'"[^"\\]*(\\.[^"\\]*)*"', '""', content)
        content = re.sub(r"'[^'\\]*(\\.[^'\\]*)*'", "''", content)

    # 2. Collapse whitespace
    content = _WHITESPACE.sub(' ', content).strip()
    return content.lower()

def tokenize_normalized(content: str, ext: str) -> List[str]:
    norm = normalize_content(content, ext)
    return re.findall(r'[a-z0-9_]+', norm)

def shingles(tokens: List[str], k: int) -> Set[str]:
    if len(tokens) < k:
        return {' '.join(tokens)} if tokens else set()
    return {' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1)}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def file_similarity(src_a: str, src_b: str, ext: str, k: int) -> float:
    ta = shingles(tokenize_normalized(src_a, ext), k)
    tb = shingles(tokenize_normalized(src_b, ext), k)
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
    """Remove a common leading directory only if all files share the same first component."""
    paths = list(files.keys())
    if not paths:
        return files
    parts = [p.split('/') for p in paths]
    first_parts = [p[0] for p in parts]
    if all(p == first_parts[0] for p in first_parts):
        # All files have the same first directory – strip it
        return { '/'.join(p[1:]): files[path] for path, p in zip(paths, parts) }
    return files

def detect_encoding(data: bytes) -> str:
    """Try to detect encoding; fallback to utf-8 with replacement."""
    try:
        import chardet
        result = chardet.detect(data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        # Some encodings may be mis‑detected; we'll try with error replacement
        data.decode(encoding, errors='replace')
        return encoding
    except ImportError:
        return 'utf-8'

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
                data = f.read()
                encoding = detect_encoding(data)
                content = data.decode(encoding, errors='replace')
                hashes.add(normalized_hash(content, ext))
    return hashes

def process_student(zip_path: Path, allowed_exts: Set[str], template_hashes: Set[str],
                    exclude_dirs: Set[str], min_lines: int) -> Student:
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
                data = f.read()
                encoding = detect_encoding(data)
                content = data.decode(encoding, errors='replace')
                all_files[info.filename] = content

    all_files = strip_common_prefix(all_files)

    student_files = {}
    for path, content in all_files.items():
        ext = path.split('.')[-1].lower()
        if normalized_hash(content, ext) not in template_hashes:
            if content.count('\n') >= min_lines:
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
            # Use word boundaries to avoid partial matches
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            counts[term] += len(pattern.findall(lower))
    return counts

# ----------------------------------------------------------------------
# Optimized Pair Comparison with Inverted Index
# ----------------------------------------------------------------------
def build_file_shingle_index(student_files: Dict[str, str], ext_map: Dict[str, str],
                             k: int) -> Dict[str, List[Tuple[str, Set[str]]]]:
    """Build a mapping from shingle to list of (file_path, shingle_set)."""
    index = defaultdict(list)
    for path, content in student_files.items():
        ext = ext_map.get(path, path.split('.')[-1].lower())
        tokens = tokenize_normalized(content, ext)
        shingle_set = shingles(tokens, k)
        for shingle in shingle_set:
            index[shingle].append((path, shingle_set))
    return index

def compare_pair(students: Tuple[Student, Student], params: Dict[str, Any]) -> dict:
    """Compare two students using an inverted index to reduce cross‑path work."""
    stud_a, stud_b = students
    file_thresh = params['file_thresh']
    cross_thresh = params['cross_thresh']
    k = params['shingle_size']

    flagged = []
    all_file_sims = []

    # Same‑path files
    common = set(stud_a.student_files.keys()) & set(stud_b.student_files.keys())
    for path in common:
        ext = path.split('.')[-1].lower()
        sim = file_similarity(stud_a.student_files[path], stud_b.student_files[path], ext, k)
        all_file_sims.append(sim)
        if sim >= file_thresh:
            flagged.append({
                'pathA': path, 'pathB': path,
                'similarity': sim,
                'contentA': stud_a.student_files[path],
                'contentB': stud_b.student_files[path],
            })

    # Cross‑path files using inverted index
    # Build extension maps
    ext_map_a = {p: p.split('.')[-1].lower() for p in stud_a.student_files}
    ext_map_b = {p: p.split('.')[-1].lower() for p in stud_b.student_files}

    # Build inverted indices: shingle -> list of (file_path, shingle_set)
    index_a = build_file_shingle_index(stud_a.student_files, ext_map_a, k)
    index_b = build_file_shingle_index(stud_b.student_files, ext_map_b, k)

    # Create fast lookup: path -> shingle_set
    path_to_shingle_a = {}
    for entries in index_a.values():
        for path, shingle_set in entries:
            path_to_shingle_a[path] = shingle_set
    path_to_shingle_b = {}
    for entries in index_b.values():
        for path, shingle_set in entries:
            path_to_shingle_b[path] = shingle_set

    # Find candidate file pairs that share at least one shingle
    common_shingles = set(index_a.keys()) & set(index_b.keys())
    candidate_pairs = set()
    for shingle in common_shingles:
        for path_a, _ in index_a[shingle]:
            for path_b, _ in index_b[shingle]:
                if path_a != path_b:
                    candidate_pairs.add((path_a, path_b))

    # Compute similarity only for candidate pairs
    for path_a, path_b in candidate_pairs:
        ext = ext_map_a[path_a]
        if ext != ext_map_b[path_b]:
            continue
        sim = jaccard(path_to_shingle_a[path_a], path_to_shingle_b[path_b])
        all_file_sims.append(sim)
        if sim >= cross_thresh:
            flagged.append({
                'pathA': path_a, 'pathB': path_b,
                'similarity': sim,
                'contentA': stud_a.student_files[path_a],
                'contentB': stud_b.student_files[path_b],
            })

    flagged.sort(key=lambda x: x['similarity'], reverse=True)

    # Overall similarity = average of top 5 file similarities
    if all_file_sims:
        top = sorted(all_file_sims, reverse=True)[:5]
        overall = sum(top) / len(top)
    else:
        overall = 0.0

    return {
        'studentA': stud_a.name,
        'studentB': stud_b.name,
        'overallSimilarity': overall,
        'flaggedFiles': flagged[:MAX_FLAGGED_FILES],
        'totalLinesA': stud_a.student_lines,
        'totalLinesB': stud_b.student_lines,
    }
# ----------------------------------------------------------------------
# Main Analysis (with progress callback)
# ----------------------------------------------------------------------
def run_analysis(template_path: Optional[Path], student_paths: List[Path],
                 allowed_exts: Set[str], term_list: List[str],
                 file_thresh: float, overall_thresh: float, cross_thresh: float,
                 exclude_dirs: Set[str], min_lines: int, shingle_size: int,
                 num_workers: int = None,
                 progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[List[Student], List[dict]]:
    if num_workers is None:
        num_workers = mp.cpu_count()

    def report(percent, msg):
        if progress_callback:
            progress_callback(percent, msg)

    report(5, "Processing template...")
    template_hashes = process_template(template_path, allowed_exts, exclude_dirs)
    print(f"Template hashes: {len(template_hashes)} unique files.")

    report(10, "Processing student ZIPs...")
    # Use process pool for I/O‑bound tasks (but ZIP reading is mostly I/O, threads are fine)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_student, p, allowed_exts, template_hashes, exclude_dirs, min_lines)
                   for p in student_paths]
        students = [f.result() for f in futures]

    report(30, "Counting terms...")
    for s in students:
        s.term_counts = check_terms(s, term_list)

    # Compare all pairs
    total_pairs = len(students)*(len(students)-1)//2
    print(f"Comparing {total_pairs} pairs using {num_workers} processes...")
    report(40, f"Comparing {total_pairs} student pairs...")

    pairs = []
    # Prepare parameters for compare_pair
    params = {
        'file_thresh': file_thresh,
        'cross_thresh': cross_thresh,
        'shingle_size': shingle_size,
    }
    # Generate all pairs (i, j)
    pair_list = [(students[i], students[j]) for i in range(len(students)) for j in range(i+1, len(students))]

    # Use process pool with tqdm progress if available
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use partial to fix params
        compare_func = partial(compare_pair, params=params)
        futures = [executor.submit(compare_func, pair) for pair in pair_list]

        if HAS_TQDM:
            iterator = tqdm(futures, total=len(pair_list), desc="Comparing pairs")
        else:
            iterator = futures

        completed = 0
        for future in iterator:
            pair = future.result()
            completed += 1
            if total_pairs > 0 and not HAS_TQDM:
                prog = 40 + int(40 * completed / total_pairs)
                report(prog, f"Comparing pairs: {completed}/{total_pairs} (found {len(pairs)} suspicious so far)")
            if pair['overallSimilarity'] >= overall_thresh:
                pairs.append(pair)

    report(80, "Sorting results...")
    pairs.sort(key=lambda x: x['overallSimilarity'], reverse=True)
    print(f"Kept {len(pairs)} pairs with overall similarity >= {overall_thresh:.2f}.")
    report(90, "Preparing report...")
    return students, pairs

# ----------------------------------------------------------------------
# HTML Report Generation (with Consolas font)
# ----------------------------------------------------------------------
def generate_html_report(students: List[Student], pairs: List[dict],
                         term_list: List[str], output_path: Path,
                         shingle_size: int, file_thresh: float, overall_thresh: float):
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

    # CSS style block
    style = """
        .diff-table, .diff-table td, .diff-table th, .diff-table pre, .diff-table code,
        .diff-container, .diff-content, .diff-header code {
            font-family: 'Cascadia Code', Consolas, 'Courier New', monospace !important;
            font-size: 9pt !important;
            line-height: 1.3 !important;
        }
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
        .similarity-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            color: white;
            font-weight: bold;
            display: inline-block;
            font-size: 0.875rem;
        }
        .card {
            transition: all 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        button.diff-toggle {
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 0.375rem;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        button.diff-toggle:hover {
            background: #2563eb;
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .file-badge-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
    """

    # Build HTML using a template string
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Checker Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        {style}
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">Plagiarism Checker Report</h1>
        <p class="text-gray-600 mb-8">Analysis of {len(students)} student submissions (shingle size={shingle_size}, file threshold={file_thresh}, overall threshold={overall_thresh})</p>
""")

    # Student overview cards
    html_parts.append("""
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Student Overview</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
""")
    for s in students:
        html_parts.append(f"""
                <div class="card bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <div class="font-medium text-gray-800">{html.escape(s.name)}</div>
                    <div class="text-sm text-gray-600 mt-2">
                        <div>Total lines: {s.total_lines}</div>
                        <div>Student lines: {s.student_lines}</div>
                        <div>Template: {s.template_pct:.1f}%</div>
                    </div>
                </div>
""")
    html_parts.append("""
            </div>
        </div>
""")

    # Term occurrences
    if term_list:
        html_parts.append("""
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Term Occurrences</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
""")
        for s in students:
            html_parts.append(f"""
                <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <div class="font-medium text-gray-800">{html.escape(s.name)}</div>
                    <div class="mt-2 flex flex-wrap gap-2">
""")
            for t in term_list:
                html_parts.append(f"""
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-200 text-gray-700">
                            {html.escape(t)}: <span class="font-bold ml-1">{s.term_counts[t]}</span>
                        </span>
""")
            html_parts.append("""
                    </div>
                </div>
""")
        html_parts.append("""
            </div>
        </div>
""")

    # Suspicious pairs
    if pairs:
        html_parts.append("""
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Suspicious Pairs</h2>
            <div class="space-y-4">
""")
        for idx, pair in enumerate(pairs):
            sim_pct = pair['overallSimilarity'] * 100
            color = sim_color(pair['overallSimilarity'])
            html_parts.append(f"""
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
""")
            for fi, f in enumerate(pair['flaggedFiles']):
                sim_pct = f['similarity'] * 100
                color = sim_color(f['similarity'])
                diff_id = f"diff_{idx}_{fi}"
                old_lines = f['contentA'].splitlines()
                new_lines = f['contentB'].splitlines()
                diff_table = html_diff.make_table(old_lines, new_lines,
                                                  fromdesc=f"{html.escape(pair['studentA'])} - {html.escape(f['pathA'])}",
                                                  todesc=f"{html.escape(pair['studentB'])} - {html.escape(f['pathB'])}",
                                                  context=False, numlines=3)
                if not diff_table.strip():
                    diff_table = '<div class="diff-empty">No differences found (files are identical).</div>'
                html_parts.append(f"""
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
""")
            html_parts.append("""
                    </div>
                </div>
""")
        html_parts.append("""
            </div>
        </div>
""")
    else:
        html_parts.append("""
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <p class="text-gray-600">No pairs above the similarity threshold.</p>
        </div>
""")

    html_parts.append("""
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
""")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))

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
    parser.add_argument('--shingle-size', type=int, default=DEFAULT_SHINGLE_SIZE,
                        help=f'Token n‑gram size (default: {DEFAULT_SHINGLE_SIZE})')
    parser.add_argument('--min-lines', type=int, default=MIN_FILE_LINES,
                        help=f'Minimum lines in a file to be considered (default: {MIN_FILE_LINES})')
    parser.add_argument('--exclude-dirs', nargs='+', default=list(EXCLUDED_DIRS),
                        help=f'Directories to exclude (default: {EXCLUDED_DIRS})')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()

    allowed_exts = set(args.exts)
    term_list = args.terms
    exclude_dirs = set(args.exclude_dirs)

    students_list, pairs = run_analysis(
        template_path=args.template,
        student_paths=args.students,
        allowed_exts=allowed_exts,
        term_list=term_list,
        file_thresh=args.file_thresh,
        overall_thresh=args.overall_thresh,
        cross_thresh=args.cross_thresh,
        exclude_dirs=exclude_dirs,
        min_lines=args.min_lines,
        shingle_size=args.shingle_size,
        num_workers=args.workers,
        progress_callback=None  # CLI could use tqdm internally
    )
    generate_html_report(students_list, pairs, term_list, args.output,
                         args.shingle_size, args.file_thresh, args.overall_thresh)
    print(f"Report generated: {args.output}")

if __name__ == '__main__':
    main()