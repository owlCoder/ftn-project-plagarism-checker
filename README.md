Plagiarism Checker
=================

A plagiarism detection tool for TypeScript, JavaScript, and Python projects using AST-based structural analysis.

Installation
------------

Install the required packages:

pip install tree-sitter tree-sitter-typescript tree-sitter-javascript tree-sitter-python

Usage
-----

python plagiarism_checker.py ./submissions ./project-template.zip

Key Features / Improvements
---------------------------

AST / Structural Analysis
- Uses tree-sitter to parse TypeScript/TSX/JS/Python into proper syntax trees.
- All identifiers are replaced with ID tokens, so similarity is 100% even if a student renames all variables and functions.

Specific Code Snippets
- A sliding-window algorithm finds the most similar 6–8 line windows across files.
- The report shows each pair of files with side-by-side comparison including line numbers.

Detection of Renamed Files
- Cross-path comparison detects cases where a student copied a file but renamed it (e.g., BookingService.ts → ReservationService.ts).
- Stricter threshold of 80% similarity for renamed files.

Configuration Options
---------------------

| Option            | Description                                           | Default |
|------------------|-------------------------------------------------------|---------|
| --threshold       | "Suspicious" similarity threshold                     | 40%     |
| --window          | AST shingle window size – smaller = more sensitive   | 8       |
| --snippet-lines   | Number of context lines in snippets                  | 8       |
| --file-threshold  | Minimum similarity for a single file to be displayed | 0.6     |

## Preview of report
<iframe src="report.html" width="100%" height="600px"></iframe>
