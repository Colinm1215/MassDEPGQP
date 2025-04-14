import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_uploads import UploadSet, configure_uploads, UploadNotAllowed
from werkzeug.utils import secure_filename
from pdfScript import get_clean_dataframe
from worker import celery_train_model, celery_process_files
import redis
from database import db, ModelSystem  # DB models
from model import create_model_record  # New function to create a model record in DB
from flask_cors import CORS
app = Flask(__name__)
app.config.from_object('Config.Config')
CORS(app, supports_credentials=True) # Allow CORS
files_upload = UploadSet('files', app.config['ALLOWED_EXTENSIONS'])
app.config['UPLOADED_FILES_DEST'] = app.config['UPLOAD_FOLDER']
configure_uploads(app, files_upload)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

with app.app_context():
    db.init_app(app)
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_last_modified(file_path):
    last_modified_timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')

def event_stream():
    pubsub = redis_client.pubsub()
    pubsub.subscribe("celery_updates")
    for message in pubsub.listen():
        if message["type"] == "message":
            yield f"data: {message['data'].decode('utf-8')}\n\n"

@app.route('/task_updates')
def task_updates():
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    files = request.files.getlist('files')
    responses = []
    for file in files:
        if file.filename == '':
            responses.append({'filename': file.filename, 'status': 'error', 'message': 'No selected file'})
        else:
            try:
                files_upload.save(file)
                responses.append({'filename': file.filename, 'status': 'success', 'message': f'"{file.filename}" uploaded successfully.'})
            except UploadNotAllowed:
                responses.append({'filename': file.filename, 'status': 'error', 'message': f'"{file.filename}" Invalid file type'})
            except Exception as e:
                responses.append({'filename': file.filename, 'status': 'error', 'message': f'"{file.filename}" {str(e)}'})
    return jsonify(responses)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/storage', methods=['GET'])
def storage():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files_list = []
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        files_list.append({"file": file, "date": get_file_last_modified(file_path)})
    return render_template('storage.html', files=files_list)

@app.route('/get_models', methods=['GET'])
def get_models():
    # Query DB for model records.
    model_records = ModelSystem.query.all()
    models_list = [record.name for record in model_records]
    return jsonify(models_list)

@app.route('/get_uploads', methods=['GET'])
def get_uploads():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify([f for f in files if allowed_file(f)])

@app.route('/get_processed_files', methods=['GET'])
def get_processed_files():
    processed_folder = app.config['PROCESSED_FOLDER']
    processed_files_list = []
    for root, _, files in os.walk(processed_folder):
        for file in files:
            if file.endswith('.csv'):
                relative_path = os.path.relpath(os.path.join(root, file), processed_folder)
                processed_files_list.append(relative_path.replace("\\", "/"))
    return jsonify(processed_files_list)

# @app.route('/model', methods=['GET'])
# def model():
#     # List uploaded CSV files.
#     # files = os.listdir(app.config['UPLOAD_FOLDER'])
#     # files_list = []
#     # for file in files:
#     #     if file.endswith('.csv'):
#     #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
#     #         files_list.append({"file": file, "date": get_file_last_modified(file_path)})
#     # Get model names from DB.
#     # model_records = ModelSystem.query.all()
#     # models_list = [record.name for record in model_records]
#     # return render_template('model.html', files=files_list, models=models_list)
#     pass  # Temporarily disabled for debugging other features

@app.route('/modelOutputs', methods=['GET'])
def modelOutputs():
    return render_template('modelOutputs.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_files():
    try:
        data = request.get_json()
        selected_files = data.get('files', [])
        if not selected_files:
            return jsonify({'status': 'error', 'message': 'No files selected'}), 400
        for file in selected_files:
            cleaned = get_clean_dataframe(file, f"uploads/{file}")
            print(f"Processing file: {file}")
            fname = file.split(".")[0]
            filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{fname}.csv")
            cleaned.to_csv(filename, index=False)
        return jsonify({'status': 'success', 'message': 'Files are being preprocessed', 'files': selected_files}), 200
    except Exception as e:
        app.logger.error(f"Error processing files: {str(e)}")
        return jsonify({"status": "error", "message": "An internal error has occurred"}), 500

@app.route('/delete', methods=['POST'])
def delete_file():
    file_path = request.form.get('file')
    file_name = secure_filename(os.path.basename(file_path))
    file_dir = os.path.dirname(file_path)
    if file_dir:
        file_path = os.path.join(app.config.get('PROCESSED_FOLDER'), file_path)
    else:
        file_path = os.path.join(app.config.get('UPLOAD_FOLDER'), file_name)
    print(file_path)
    if not file_name:
        return jsonify({"status": "error", "message": "No file name provided"}), 400
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"status": "success", "message": "File deleted successfully"})
        except Exception as e:
            app.logger.error(f"Error deleting file {file_name}: {str(e)}")
            return jsonify({"status": "error", "message": "An internal error has occurred"}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404

@app.route('/deleteMultiple', methods=['POST'])
def delete_files():
    data = request.get_json()
    file_names = data.get('files')
    if not file_names:
        return jsonify({"status": "error", "message": "No files provided"}), 400
    failed_files = []
    success_files = []
    for file_path in file_names:
        file_name = secure_filename(os.path.basename(file_path))
        file_dir = os.path.dirname(file_path)
        if file_dir:
            file_path = os.path.join(app.config.get('PROCESSED_FOLDER'), file_path)
        else:
            file_path = os.path.join(app.config.get('UPLOAD_FOLDER'), file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                success_files.append(file_name)
            except Exception as e:
                app.logger.error(f"Error deleting file {file_name}: {str(e)}")
                failed_files.append({"file": file_name, "error": "An internal error has occurred"})
        else:
            failed_files.append({"file": file_name, "error": "File not found"})
    result = {"status": "success", "message": "Files deleted successfully", "deleted_files": success_files} if success_files else {}
    if failed_files:
        result = {"status": "error", "message": "Some files could not be deleted", "failed_files": failed_files}
    if not result:
        result = {"status": "error", "message": "No files were deleted"}
    return jsonify(result)

@app.route('/create_model', methods=['POST'])
def create_model():
    data = request.get_json()
    model_name = data.get('model_name', '').strip()
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    # Check if model exists in DB.
    existing = ModelSystem.query.filter_by(name=model_name).first()
    if existing:
        return jsonify({"error": f"Model {model_name} already exists"}), 400
    new_model = create_model_record(model_name)
    if new_model:
        return jsonify({"message": f"Model {model_name} created"}), 200
    else:
        return jsonify({"error": f"Failed to create model: {model_name}"}), 400

def model_validation(model_name):
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    existing = ModelSystem.query.filter_by(name=model_name).first()
    if not existing:
        return jsonify({"error": "Model could not be validated"}), 400
    return None

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    primary_files = data.get("primary_files", [])
    if not primary_files:
        return jsonify({"error": "No training files provided"}), 400
    secondary_files = data.get("secondary_files", [])
    if not secondary_files:
        return jsonify({"error": "No standardized files provided"}), 400
    model_name = data.get('model_name')
    bypass_already_trained = data.get('bypass_already_trained')
    valid = model_validation(model_name)
    if valid:
        return valid
    task = celery_train_model.delay(primary_files, secondary_files, model_name, bypass_already_trained)
    return jsonify({"message": f"Model training started for {model_name}", "task_id": task.id}), 200

@app.route('/process_files', methods=['POST'])
def process_files():
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    valid = model_validation(model_name, file_names)
    if valid:
        return valid
    task = celery_process_files.delay(file_names, model_name)
    return jsonify({"message": "File processing started.", "task_id": task.id}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
