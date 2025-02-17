import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_uploads import UploadSet, configure_uploads, UploadNotAllowed
from werkzeug.utils import secure_filename

from model import create_model_pkl
from pdfScript import get_clean_dataframe
from worker import celery_train_model, celery_process_files
import redis


app = Flask(__name__)
app.config.from_object('Config.Config')
files_upload = UploadSet('files', app.config['ALLOWED_EXTENSIONS'])
app.config['UPLOADED_FILES_DEST'] = app.config['UPLOAD_FOLDER']
configure_uploads(app, files_upload)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_file_last_modified(file_path):
    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_time = datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return last_modified_time

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
            responses.append({
                'filename': file.filename,
                'status': 'error',
                'message': 'No selected file'
            })
        else:
            try:
                files_upload.save(file)
                responses.append({
                    'filename': file.filename,
                    'status': 'success',
                    'message': f'"{file.filename}" uploaded successfully.'})
            except UploadNotAllowed:
                responses.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': f'"{file.filename}" Invalid file type'
                })
            except Exception as e:
                responses.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': f'"{file.filename}" {str(e)}'})

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
        last_modified_date = get_file_last_modified(file_path)
        f = {"file": file, "date": last_modified_date}
        files_list.append(f)
    return render_template('storage.html', files=files_list)

@app.route('/get_models', methods=['GET'])
def get_models():
    models = os.listdir(app.config['MODEL_FOLDER'])
    models_list = [m for m in models if m.endswith('.pkl')]
    return jsonify(models_list)

@app.route('/model', methods=['GET'])
def model():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files_list = []
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            last_modified_date = get_file_last_modified(file_path)
            f = {"file": file, "date": last_modified_date}
            files_list.append(f)
    models = os.listdir(app.config['MODEL_FOLDER'])
    print(models)
    models_list = []
    for m in models:
        if m.endswith('.pkl'):
            models_list.append(m)
    print(models_list)
    return render_template('model.html', files=files_list, models=models_list)


@app.route('/preprocess', methods=['POST'])
def preprocess_files():
    try:
        data = request.get_json()
        selected_files = data.get('files', [])

        if not selected_files:
            return jsonify({'status': 'error', 'message': 'No files selected'})

        for file in selected_files:
            cleaned = get_clean_dataframe(file, f"uploads/{file}")
            print(f"Processing file: {file}")
            fname = file.split(".")[0]
            filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{fname}.csv")
            cleaned.to_csv(filename, index=False)

        return jsonify({'status': 'success', 'message': 'Files are being preprocessed', 'files': selected_files})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete', methods=['POST'])
def delete_file():
    file_name = request.form.get('file')

    if not file_name:
        return jsonify({"status": "error", "message": "No file name provided"}), 400

    upload_folder = app.config.get('UPLOAD_FOLDER')
    if not upload_folder:
        return jsonify({"status": "error", "message": "Upload folder not set"}), 500

    file_path = os.path.join(upload_folder, file_name)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"status": "success", "message": "File deleted successfully"})
        except Exception as e:
            return jsonify({"status": "error", "message": f'{str(e)} 111'}), 500
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404


@app.route('/deleteMultiple', methods=['POST'])
def delete_files():
    data = request.get_json()
    file_names = data.get('files')
    print(file_names)

    if not file_names:
        return jsonify({"status": "error", "message": "No files provided"}), 400

    upload_folder = app.config.get('UPLOAD_FOLDER')
    if not upload_folder:
        return jsonify({"status": "error", "message": "Upload folder not set"}), 500

    failed_files = []
    success_files = []

    for file_name in file_names:
        file_path = os.path.join(upload_folder, file_name)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                success_files.append(file_name)
            except Exception as e:
                failed_files.append({"file": file_name, "error": str(e)})
        else:
            failed_files.append({"file": file_name, "error": "File not found"})

    result = None
    if success_files:
        result = {"status": "success", "message": "Files deleted successfully", "deleted_files": success_files}
    if failed_files:
        result = {"status": "error", "message": "Some files could not be deleted", "failed_files": failed_files}

    if not result:
        result = {"status": "error", "message": "No files were deleted"}

    return jsonify(result)

@app.route('/create_model', methods=['POST'])
def create_model():
    data = request.get_json()
    model_name = secure_filename(data.get('model_name'))

    r = os.path.join(app.config['MODEL_FOLDER'], model_name)

    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    if not model_name.endswith(".pkl"):
        return jsonify({"error": "Invalid model type (file is not .pkl)"}), 400
    if os.path.exists(r):
        return jsonify({"error": f"Model {model_name} already exists"}), 400
    print(f"Model type: {type(model_name)}")
    r = secure_filename(create_model_pkl(model_name))
    if os.path.exists(r):
        return jsonify({"message": f"Model {model_name} created"}), 200
    else:
        return jsonify({"error": f"Failed to create model: {r}"}), 400

def model_validation(model_name, file_names):
    if not file_names:
        return jsonify({"error": "No files provided"}), 400
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    secure_name = secure_filename(model_name)
    model_path = os.path.join("models", secure_name)
    if not model_name.endswith(".pkl"):
        return jsonify({"error": "Invalid model type (file is not .pkl)"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": "Model could not be validated"}), 400
    return None
@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    bypass_already_trained = data.get('bypass_already_trained')
    valid = model_validation(model_name, file_names)
    if valid:
        return valid, 400

    #threading.Thread(target=modelClass.train, args=(pdf_dfs, model_name, update_training_status, bypass_already_trained)).start()
    task = celery_train_model.delay(file_names, model_name, bypass_already_trained)
    return jsonify({"message": f"Model training started for {model_name}", "task_id": task.id}), 200

@app.route('/process_files', methods=['POST'])
def process_files():
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    valid = model_validation(model_name, file_names)
    if valid:
        return valid, 400

    #def process_helper():
    #    error, standardized_dfs = modelClass.process_pdfs(pdf_dfs, model_name, update_process_status)
    #    if standardized_dfs is not None:
    #        print(standardized_dfs)
    #        for df in standardized_dfs:
    #            save_standardized_reports(df)

    #threading.Thread(target=process_helper).start()
    task = celery_process_files.delay(file_names, model_name)
    return jsonify({"message": "File processing started.", "task_id": task.id}), 200


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
