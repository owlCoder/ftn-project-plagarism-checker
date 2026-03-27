import tempfile
import threading
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import our plagiarism analysis functions (updated version with optional template)
from plagiarism_analyzer import generate_html_report, run_analysis, EXCLUDED_DIRS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # change in production
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['RESULT_FOLDER'] = tempfile.mkdtemp()

# In-memory store for analysis results (session_id -> result html path or error)
results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get form data
    template_file = request.files.get('template')   # may be None
    student_files = request.files.getlist('students')
    exts = request.form.get('extensions', 'py,js,ts,java,cpp,c,cs,sql')
    threshold = float(request.form.get('threshold', 0.4))
    file_threshold = float(request.form.get('file_threshold', 0.6))
    terms = request.form.get('terms', 'any,exception,todo')
    exclude_dirs = request.form.get('exclude_dirs', ','.join(EXCLUDED_DIRS))

    if not student_files:
        return "At least one student file is required", 400

    # Save uploaded files to temporary location
    session_id = str(uuid.uuid4())
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save template if provided
    template_path = None
    if template_file and template_file.filename:
        template_path = upload_dir / secure_filename(template_file.filename)
        template_file.save(template_path)
        logging.info(f"Template saved: {template_path}")

    student_paths = []
    for f in student_files:
        if f and allowed_file(f.filename):
            path = upload_dir / secure_filename(f.filename)
            f.save(path)
            student_paths.append(path)
            logging.info(f"Student file saved: {path}")

    # Store settings for later use in background task
    analysis_params = {
        'session_id': session_id,
        'template_path': template_path,
        'student_paths': student_paths,
        'exts': set(exts.split(',')),
        'threshold': threshold,
        'file_threshold': file_threshold,
        'terms': [t.strip() for t in terms.split(',') if t.strip()],
        'exclude_dirs': set(exclude_dirs.split(',')),
    }

    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis_async, args=(analysis_params,))
    thread.daemon = True
    thread.start()

    return jsonify({'session_id': session_id, 'status': 'processing'})

def run_analysis_async(params):
    """Run the analysis and store the result HTML path."""
    session_id = params['session_id']
    template_path = params['template_path']
    student_paths = params['student_paths']
    exts = params['exts']
    threshold = params['threshold']
    file_threshold = params['file_threshold']
    terms = params['terms']
    exclude_dirs = params['exclude_dirs']

    try:
        students, pairs = run_analysis(
            template_path=template_path,
            student_paths=student_paths,
            allowed_exts=exts,
            term_list=terms,
            file_thresh=file_threshold,
            overall_thresh=threshold,
            cross_thresh=0.3,
            exclude_dirs=exclude_dirs,
            num_workers=None
        )
        # Generate HTML report
        result_dir = Path(app.config['RESULT_FOLDER']) / session_id
        result_dir.mkdir(parents=True, exist_ok=True)
        report_path = result_dir / 'report.html'
        generate_html_report(students, pairs, terms, report_path)
        results[session_id] = str(report_path)
        logging.info(f"Analysis completed for session {session_id}, report at {report_path}")
    except Exception as e:
        logging.exception("Analysis failed")
        results[session_id] = f"ERROR: {str(e)}"

@app.route('/status/<session_id>')
def status(session_id):
    """Poll for analysis completion."""
    if session_id not in results:
        return jsonify({'status': 'processing'})
    result = results[session_id]
    if result.startswith('ERROR'):
        return jsonify({'status': 'error', 'message': result})
    else:
        return jsonify({'status': 'done', 'report_url': url_for('report', session_id=session_id)})

@app.route('/report/<session_id>')
def report(session_id):
    """Display the generated report."""
    report_path = results.get(session_id)
    if not report_path or report_path.startswith('ERROR'):
        return "Report not found", 404
    with open(report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return "File too large. Max size 500MB.", 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)