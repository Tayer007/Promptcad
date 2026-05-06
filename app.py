"""
PromptCAD - Local inference webapp
"""
import os, sys, json, uuid, threading, subprocess, time
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from enhancer import run_enhancement

app = Flask(__name__, static_folder='static')
CORS(app)

BASE_DIR    = os.path.dirname(__file__)
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PYTHON           = sys.executable
INFERENCE_SCRIPT = os.path.join(BASE_DIR, 'inference.py')

# ── SolidWorks Builder executable ──────────────────────────────────────────
# Set this to the path of your compiled Builder.exe.
# The Builder is called as:  Builder.exe <path_to_translated.json>
BUILDER_EXE = r"C:\Users\fou0\Downloads\code\v6\Builder\bin\Release\net6.0\Builder.exe"

jobs = {}

def run_inference_local(job_id, stl_path):
    job = jobs[job_id]
    stl_out       = os.path.join(RESULTS_DIR, f"{job_id}_generated.stl")
    py_out        = os.path.join(RESULTS_DIR, f"{job_id}_generated.py")
    corrected_stl = os.path.join(RESULTS_DIR, f"{job_id}_corrected.stl")
    corrected_py  = os.path.join(RESULTS_DIR, f"{job_id}_corrected.py")
    result_json   = os.path.join(RESULTS_DIR, f"{job_id}_result.json")

    try:
        job['steps']['inference']['status'] = 'running'
        job['steps']['inference']['message'] = 'Loading model onto GPU...'

        proc = subprocess.Popen(
            [PYTHON, INFERENCE_SCRIPT,
             stl_path, stl_out, py_out,
             corrected_stl, corrected_py, result_json],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

        log_lines = []
        scaling_started = False
        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            print(line)
            if 'Model loaded' in line:
                job['steps']['inference']['message'] = 'Model loaded — generating candidates...'
            elif 'Sample ' in line:
                n = line.split('Sample ')[1].split(':')[0].strip()
                job['steps']['inference']['message'] = f'Generated sample {n}/10...'
            elif 'Running SmartScaler' in line:
                job['steps']['inference']['status'] = 'done'
                job['steps']['inference']['message'] = 'Inference complete'
                job['steps']['scale']['status'] = 'running'
                job['steps']['scale']['message'] = 'Analyzing dimensions...'
                scaling_started = True
            elif 'Scale applied' in line and scaling_started:
                job['steps']['scale']['message'] = line.strip()

        proc.wait()
        if proc.returncode != 0:
            raise Exception('\n'.join(log_lines[-10:]))

        job['steps']['inference']['status'] = 'done'
        job['steps']['scale']['status'] = 'done'
        job['steps']['scale']['message'] = 'Rescaling complete'

        if not os.path.exists(result_json):
            raise Exception('Result JSON not found')

        with open(result_json) as f:
            output = json.load(f)

        si = output.get('scale_info', {})
        job['status'] = 'done'
        job['result'] = {
            'generated_code':    output.get('generated_code', ''),
            'corrected_code':    output.get('corrected_code', ''),
            'chamfer_distance':  output.get('chamfer_distance'),
            'gen_stl_url':       f'/results/{job_id}_generated.stl',
            'corrected_stl_url': f'/results/{job_id}_corrected.stl' if os.path.exists(corrected_stl) else None,
            'scale_info': {
                'scale_x':       si.get('scale_x', 1.0),
                'scale_y':       si.get('scale_y', 1.0),
                'scale_z':       si.get('scale_z', 1.0),
                'mode':          si.get('mode', 'unknown'),
                'is_uniform':    si.get('is_uniform', True),
                'uniform_scale': si.get('uniform_scale', 1.0),
                'gt_extents':    si.get('gt_extents', []),
                'gen_extents':   si.get('gen_extents', []),
                'llm_used':      si.get('llm_used', False),
                'error':         si.get('error'),
            },
        }

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        for step in job['steps'].values():
            if step['status'] == 'running':
                step['status'] = 'error'
                step['message'] = str(e)[:200]


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.stl'):
        return jsonify({'error': 'Only STL files accepted'}), 400
    job_id   = str(uuid.uuid4())[:8]
    stl_path = os.path.join(UPLOAD_DIR, f"{job_id}.stl")
    file.save(stl_path)
    jobs[job_id] = {
        'id': job_id, 'status': 'running', 'filename': file.filename,
        'stl_url': f'/uploads/{job_id}.stl',
        'stl_path': stl_path,
        'steps': {
            'inference': {'status': 'pending', 'message': 'Waiting...'},
            'scale':     {'status': 'pending', 'message': 'Waiting...'},
        },
        'result': None, 'error': None,
    }
    t = threading.Thread(target=run_inference_local, args=(job_id, stl_path))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id, 'stl_url': f'/uploads/{job_id}.stl'})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/enhance/<job_id>', methods=['POST'])
def enhance(job_id):
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        job = jobs[job_id]
        if job['status'] != 'done':
            return jsonify({'error': 'Job not complete yet'}), 400

        data = request.get_json(force=True, silent=True) or {}
        orig_b64   = data.get('original_img', '')
        corr_b64   = data.get('corrected_img', '')
        model      = data.get('model', 'qwen3.5:122b')
        ollama_url = data.get('ollama_url', 'http://127.0.0.1:11434')

        # Strip data-URL prefix if the browser included it
        if ',' in orig_b64:
            orig_b64 = orig_b64.split(',', 1)[1]
        if ',' in corr_b64:
            corr_b64 = corr_b64.split(',', 1)[1]

        job['enhance_status']  = 'starting'
        job['enhance_message'] = 'Starting supervisor correction…'
        job['enhance_result']  = None

        t = threading.Thread(
            target=run_enhancement,
            args=(job_id, jobs, RESULTS_DIR, orig_b64, corr_b64, model, ollama_url),
        )
        t.daemon = True
        t.start()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'started'})

@app.route('/enhance_status/<job_id>')
def enhance_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'status':  job.get('enhance_status', 'idle'),
        'message': job.get('enhance_message', ''),
        'result':  job.get('enhance_result'),
    })

@app.route('/enhance_stream/<job_id>')
def enhance_stream(job_id):
    @stream_with_context
    def generate():
        last_status = None
        while True:
            if job_id not in jobs:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                break
            job = jobs[job_id]
            status  = job.get('enhance_status', 'idle')
            message = job.get('enhance_message', '')
            result  = job.get('enhance_result')
            payload = {'status': status, 'message': message, 'result': result}
            # Always send on first tick, then only when something changed
            if status != last_status or status in ('done', 'error'):
                yield f"data: {json.dumps(payload)}\n\n"
                last_status = status
            if status in ('done', 'error'):
                break
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ─────────────────────────────────────────────────────────────────────────────
# Translate & Export to SolidWorks
# ─────────────────────────────────────────────────────────────────────────────

def run_export_job(job_id: str, jobs: dict, results_dir: str, use_supervised: bool):
    """
    Background thread:
      1. Pick best available code (supervised > corrected > generated).
      2. Run cq_to_sw_json.translate() → save {job_id}_translated.json.
      3. Launch Builder with os.startfile — identical to double-clicking the exe.
    """
    job = jobs[job_id]

    json_path = os.path.join(results_dir, f"{job_id}_translated.json")
    actual_dir = os.path.join(results_dir, 'actual')
    os.makedirs(actual_dir, exist_ok=True)
    actual_json_path = os.path.join(actual_dir, f"{job_id}_translated.json")
    job['export_json_path'] = json_path

    try:
        # ── 1. Translate ──────────────────────────────────────
        job['export_status']  = 'translating'
        job['export_message'] = 'Translating CadQuery → SolidWorks JSON…'

        if use_supervised:
            code = (job.get('enhance_result') or {}).get('code', '')
        else:
            code = None

        if not code:
            code = (
                job.get('result', {}).get('corrected_code') or
                job.get('result', {}).get('generated_code', '')
            )

        if not code:
            raise ValueError("No CadQuery code available — run inference first")

        sys.path.insert(0, BASE_DIR)
        from cq_to_sw_json import translate as cq_translate  # noqa: PLC0415

        translated = cq_translate(code)
        n_ops = len(translated.get('operations', []))

        json_str = json.dumps(translated, indent=2)
        with open(json_path, 'w') as f:
            f.write(json_str)
        with open(actual_json_path, 'w') as f:
            f.write(json_str)

        job['export_message'] = f'Translation done — {n_ops} operation(s). Launching Builder…'

        # ── 2. Launch Builder ─────────────────────────────────
        job['export_status']  = 'building'
        job['export_message'] = f'Launching SolidWorks Builder ({n_ops} ops)…'

        if not os.path.exists(BUILDER_EXE):
            job['export_status']  = 'done'
            job['export_message'] = (
                f'Translation saved ({n_ops} ops). '
                f'Builder not found — set BUILDER_EXE in app.py.'
            )
            return

        proc = subprocess.Popen(
            [BUILDER_EXE, actual_json_path],
            cwd=os.path.dirname(BUILDER_EXE),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            close_fds=True,
        )
        print(f"[builder] launched PID={proc.pid}", flush=True)

        job['export_status']  = 'done'
        job['export_message'] = f'Builder gestartet (PID {proc.pid}) — SolidWorks öffnet das Modell.'

    except Exception as e:
        job['export_status']  = 'error'
        job['export_message'] = str(e)[:400]


@app.route('/export/<job_id>', methods=['POST'])
def export_to_sw(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': 'Inference not complete yet'}), 400

    data           = request.get_json(force=True, silent=True) or {}
    use_supervised = bool(data.get('use_supervised', False))

    job['export_status']  = 'starting'
    job['export_message'] = 'Starting export…'
    job['export_json_path'] = ''

    t = threading.Thread(
        target=run_export_job,
        args=(job_id, jobs, RESULTS_DIR, use_supervised),
    )
    t.daemon = True
    t.start()

    return jsonify({'status': 'started'})


@app.route('/export_status/<job_id>')
def export_status_route(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'status':    job.get('export_status',   'idle'),
        'message':   job.get('export_message',  ''),
        'json_path': job.get('export_json_path', ''),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
