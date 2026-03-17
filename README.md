Plagiarism Checker
=================

A plagiarism detection tool for student code submissions with **GitHub-style diff visualization** for suspicious files. Supports TypeScript, JavaScript, Python, Java, C++, C#, and more.

Installation
------------

Install the required packages:

```bash
pip install tree-sitter tree-sitter-typescript tree-sitter-javascript tree-sitter-python
```

Usage
-----
```bash
python plagiarism_checker.py ./submissions ./project-template.zip
```

Full command with options:
```bash
python plagiarism_checker.py ./submissions ./template.zip \
  --threshold 50 \
  --file-threshold 0.6 \
  --output report.html \
  --extensions ts,tsx,js,jsx,py,java,cpp,c,h,cs
```

Key Features
---------------------------

### Template Code Subtraction
- Automatically removes unchanged template/starter code from analysis
- Only student-written code is compared
- Detects both exact matches and renamed template files

### Multi-Level Similarity Detection
- **Overall pair similarity**: Weighted Jaccard index across all student files
- **Per-file matching**: Identifies specific files with high similarity
- **Cross-path detection**: Catches renamed files (e.g., `BookingService.ts` → `ReservationService.ts`)

### Diff Viewer
- **Side-by-side code comparison** for all suspicious file pairs
- **Syntax highlighting** with language detection
- **Line-by-line diff** showing additions, deletions, and unchanged code
- **Expandable sections** to focus on specific files
- **Color-coded similarity** (green → yellow → red)

### Interactive HTML Report
- **Live filtering** by similarity threshold
- **Summary statistics** with visual indicators
- **Student overview** showing template vs. original code percentages
- **Expandable pair cards** with detailed file-by-file breakdown
- **Dark mode** professional design

### Advanced Analysis
- **Normalized comparison**: Strips comments, whitespace, and string literals
- **Token-based shingling** (k-gram analysis)
- **Jaccard similarity** metric
- **Multiple file format support**: ts, tsx, js, jsx, py, java, cs, cpp, c, h, sql

Report Structure
----------------

The generated HTML report contains:

1. **Summary Dashboard**
   - Total suspicious pairs (>40% similarity)
   - Average template code percentage
   - Maximum similarity found

2. **Student Overview Table**
   - Total lines of code per student
   - Student-written lines (after template subtraction)
   - Template percentage

3. **Pairwise Comparison Section**
   - Interactive slider to filter pairs by similarity
   - Expandable cards for each pair showing:
     - Overall similarity score
     - Number of flagged files
     - Student line counts
   
4. **File-Level Details** (NEW)
   - GitHub-style diff for each suspicious file pair
   - Syntax-highlighted side-by-side view
   - Line numbers on both sides
   - Color coding: green (additions), red