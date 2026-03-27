# Plagiarism Checker

A powerful plagiarism detection tool for student code submissions that subtracts template code and visualizes suspicious similarities with **GitHub-style diff views**.

## Features

- **Template code subtraction** – automatically removes unchanged template/starter code
- **Multi-language support** – TypeScript, JavaScript, Python, Java, C++, C#, SQL, and more
- **Per‑file and cross‑file similarity** – catches renamed files (e.g., `BookingService.ts` → `ReservationService.ts`)
- **Interactive web interface** – upload files, configure settings, and view results in real‑time
- **Light‑mode diff viewer** – clear line‑by‑line additions, deletions, and unchanged code
- **Parallel processing** – uses all CPU cores for fast analysis
- **Directory exclusion** – ignore common framework folders like `dist`, `__pycache__`, `node_modules`, etc.
- **Configurable** – choose file extensions, similarity thresholds, terms to search, and excluded directories

## Installation

```bash
# Clone the repository
git clone https://github.com/owlCoder/ftn-project-plagarism-checker
cd plagiarism-checker

# Install required packages
pip install flask xxhash
```

*Optional*: For faster hashing, `xxhash` is recommended. Without it, the tool falls back to MD5 (still fast).

## Usage

### Web Interface (Recommended)

Start the Flask application:

```bash
python main.py
```

Open your browser at `http://localhost:5000`. Upload:

- **Template ZIP** – contains the starter code
- **Student ZIPs** – one or more student submissions (all `.zip` files)

Adjust settings:

- **File extensions** – comma‑separated extensions to analyze (default: `py,js,ts,java,cpp,c,cs,sql`)
- **Overall similarity threshold** – minimum similarity to report a pair (0‑1, default: 0.4)
- **Per‑file similarity threshold** – threshold for flagging individual files (0‑1, default: 0.6)
- **Terms to check** – words to search for (e.g., `any,exception,todo`)
- **Excluded directories** – folders to ignore (case‑insensitive, default: `dist-electron,dist,public,__pycache__,node_modules,.git,build,target,venv,env,.idea,.vscode`)

Click **Run Analysis**. The results will appear in a new tab when processing finishes.

### Command‑Line Tool

If you prefer the command line, you can also run:

```bash
python plagiarism_checker.py template.zip submissions/ --output report.html
```

**Command‑line options**:

| Option | Description |
|--------|-------------|
| `template` | Path to the template ZIP file |
| `students_folder` | Folder containing student submission ZIPs |
| `--output` | Output HTML report file (default: `report.html`) |
| `--exts` | Comma‑separated file extensions to consider (default: `py,js,ts,java,cpp,c,cs,sql`) |
| `--threshold` | Overall similarity threshold (0‑1, default: 0.4) |
| `--file-threshold` | Per‑file similarity threshold (0‑1, default: 0.6) |
| `--terms` | Terms to check for (comma‑separated, default: `any,exception,todo`) |
| `--exclude-dirs` | Directories to ignore (comma‑separated, default: `dist-electron,dist,public,__pycache__,node_modules,.git,build,target,venv,env,.idea,.vscode`) |
| `--workers` | Number of worker threads (default: CPU count) |

## How It Works

### Template Code Subtraction
- Each file in the student ZIP is compared to the template using a normalized hash (comments and strings removed).
- Only files that differ from the template are kept for further analysis.

### Similarity Detection
- **Overall student similarity** – uses TF‑IDF with cosine similarity to compare the combined code of two students.
- **Per‑file similarity** – uses MinHash signatures (128 hash functions) to quickly estimate Jaccard similarity between individual files.
- **Cross‑path detection** – also compares files with different names, catching renamed or restructured files.

### Diff Viewer
- For every suspicious file pair (similarity ≥ file threshold), the report shows a side‑by‑side diff.
- Uses `difflib.HtmlDiff` to generate GitHub‑style tables with:
  - Line numbers on both sides
  - Green background for additions
  - Red background for deletions
  - Light background for unchanged code
- Diffs are hidden by default and can be expanded with a button.

### HTML Report
- **Student overview** – total lines, student‑written lines, template percentage.
- **Term occurrences** – counts of user‑defined keywords (e.g., "any", "exception").
- **Suspicious pairs** – cards that expand to show all flagged files, each with its own diff toggle.
- **Interactive slider** to filter pairs by overall similarity (in the web interface).
- **Color‑coded similarity badges** (green → yellow → red).

### Performance Optimizations
- **Caching** – normalized content and hashes are cached to avoid recomputation.
- **MinHash** – reduces per‑file similarity calculation to a fixed number of operations (128 per file).
- **Parallelism** – uses `ThreadPoolExecutor` for I/O‑bound tasks and `ProcessPoolExecutor` for CPU‑bound tasks.
- **Pre‑filtering** – ignores files with very few tokens.

## Directory Exclusion

The tool automatically skips common build and dependency directories (case‑insensitive). You can customize the list in the settings or via the `--exclude-dirs` flag. This prevents noise from files like:

- `dist/`, `build/`, `target/` – compiled outputs
- `node_modules/`, `venv/`, `env/` – dependencies
- `__pycache__/`, `.git/`, `.idea/` – system folders

## Example Output

After analysis, you'll receive an HTML report similar to:

- **Top section** – summary cards with total suspicious pairs and average template percentage.
- **Student Overview** – a table showing each student's lines of code, student‑written lines, and template share.
- **Suspicious Pairs** – expandable cards for each pair. Click to see a list of flagged files.
- **File Diffs** – click **Show Diff** to view side‑by‑side comparisons with syntax‑like coloring.

## Contributing

Contributions are welcome! Please open an issue or pull request for any bugs or enhancements.
