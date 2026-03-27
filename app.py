import tempfile
import threading
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

logging.basicConfig(level=logging.INFO)

# Import the improved analyzer
from plagiarism_analyzer import generate_html_report, run_analysis, EXCLUDED_DIRS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['RESULT_FOLDER'] = tempfile.mkdtemp()

results = {}
progress_store = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    template_file = request.files.get('template')
    student_files = request.files.getlist('students')
    # ----- existing parameters -----
    exts = request.form.get('extensions', 'py,js,ts,java,cpp,c,cs,sql')
    overall_thresh = float(request.form.get('overall_threshold', 0.4))   # renamed for clarity
    file_thresh = float(request.form.get('file_threshold', 0.6))
    terms = request.form.get('terms', 'any,exception,todo')
    exclude_dirs = request.form.get('exclude_dirs', ','.join(EXCLUDED_DIRS))
    # ----- new parameters -----
    shingle_size = int(request.form.get('shingle_size', 5))
    min_lines = int(request.form.get('min_lines', 10))
    cross_thresh = float(request.form.get('cross_threshold', 0.7))

    if not student_files:
        return "At least one student file is required", 400

    session_id = str(uuid.uuid4())
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    template_path = None
    if template_file and template_file.filename:
        template_path = upload_dir / secure_filename(template_file.filename)
        template_file.save(template_path)

    student_paths = []
    for f in student_files:
        if f and allowed_file(f.filename):
            path = upload_dir / secure_filename(f.filename)
            f.save(path)
            student_paths.append(path)

    analysis_params = {
        'session_id': session_id,
        'template_path': template_path,
        'student_paths': student_paths,
        'exts': set(exts.split(',')),
        'overall_thresh': overall_thresh,
        'file_thresh': file_thresh,
        'cross_thresh': cross_thresh,
        'terms': [t.strip() for t in terms.split(',') if t.strip()],
        'exclude_dirs': set(exclude_dirs.split(',')),
        'shingle_size': shingle_size,
        'min_lines': min_lines,
    }

    progress_store[session_id] = {'status': 'processing', 'progress': 0, 'message': 'Starting...'}

    thread = threading.Thread(target=run_analysis_async, args=(analysis_params,))
    thread.daemon = True
    thread.start()

    return jsonify({'session_id': session_id, 'status': 'processing'})

def run_analysis_async(params):
    session_id = params['session_id']
    template_path = params['template_path']
    student_paths = params['student_paths']
    exts = params['exts']
    overall_thresh = params['overall_thresh']
    file_thresh = params['file_thresh']
    cross_thresh = params['cross_thresh']
    terms = params['terms']
    exclude_dirs = params['exclude_dirs']
    shingle_size = params['shingle_size']
    min_lines = params['min_lines']

    def update_progress(percent, message):
        if session_id in progress_store:
            progress_store[session_id]['progress'] = percent
            progress_store[session_id]['message'] = message

    try:
        update_progress(5, "Processing template...")
        students, pairs = run_analysis(
            template_path=template_path,
            student_paths=student_paths,
            allowed_exts=exts,
            term_list=terms,
            file_thresh=file_thresh,
            overall_thresh=overall_thresh,
            cross_thresh=cross_thresh,
            exclude_dirs=exclude_dirs,
            min_lines=min_lines,
            shingle_size=shingle_size,
            num_workers=None,          # let the analyzer decide (CPU count)
            progress_callback=update_progress
        )
        update_progress(85, "Generating HTML report...")
        result_dir = Path(app.config['RESULT_FOLDER']) / session_id
        result_dir.mkdir(parents=True, exist_ok=True)
        report_path = result_dir / 'report.html'
        generate_html_report(
            students, pairs, terms, report_path,
            shingle_size=shingle_size,
            file_thresh=file_thresh,
            overall_thresh=overall_thresh
        )
        results[session_id] = str(report_path)
        update_progress(100, "Done")
        progress_store[session_id]['status'] = 'done'
        logging.info(f"Analysis completed for session {session_id}")
    except Exception as e:
        logging.exception("Analysis failed")
        results[session_id] = f"ERROR: {str(e)}"
        progress_store[session_id]['status'] = 'error'
        progress_store[session_id]['message'] = f"Error: {str(e)}"

@app.route('/status/<session_id>')
def status(session_id):
    if session_id not in progress_store:
        return jsonify({'status': 'processing', 'progress': 0, 'message': 'Initializing...'})
    progress = progress_store[session_id]
    if progress['status'] == 'done':
        return jsonify({
            'status': 'done',
            'report_url': url_for('report', session_id=session_id),
            'progress': 100,
            'message': 'Complete'
        })
    elif progress['status'] == 'error':
        return jsonify({
            'status': 'error',
            'message': progress.get('message', 'Unknown error'),
            'progress': 0
        })
    else:
        return jsonify({
            'status': 'processing',
            'progress': progress.get('progress', 0),
            'message': progress.get('message', 'Processing...')
        })

@app.route('/report/<session_id>')
def report(session_id):
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