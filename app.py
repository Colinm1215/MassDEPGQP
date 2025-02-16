import os
import threading
import time
from datetime import datetime

from flask import Flask, request, jsonify, render_template, Response

from model import PDFStandardizer, create_model_pkl
from pdfScript import get_clean_dataframe

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
modelClass = PDFStandardizer()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_last_modified(file_path):
    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_time = datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return last_modified_time


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
            cleaned = get_clean_dataframe(f"uploads/{file}")
            print(f"Processing file: {file}")
            filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{file.split(".")[0]}.csv")
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


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    responses = []

    for file in files:
        if file.filename == '':
            response = {'filename': file.filename, 'status': 'error', 'message': 'No selected file'}
        elif allowed_file(file.filename):
            try:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                response = {'filename': file.filename, 'status': 'success',
                            'message': f'"{file.filename}" uploaded successfully.'}
            except Exception as e:
                response = {'filename': file.filename, 'status': 'error', 'message': f'"{file.filename}" {str(e)}'}
        elif not allowed_file(file.filename):
            response = {'filename': file.filename, 'status': 'error', 'message': f'"{file.filename}" Invalid file type'}
        else:
            response = {'filename': file.filename, 'status': 'error', 'message': f'"{file.filename}" Unknown Error'}
        print(response)
        responses.append(response)

    return jsonify(responses)


training_updates = []
process_updates = []


def update_training_status(message):
    training_updates.append(message)


def update_process_status(message):
    process_updates.append(message)


def generate_training_output():
    global training_updates
    while True:
        if training_updates:
            yield f"data: {training_updates.pop(0)}\n\n"
        time.sleep(1)


def generate_process_output():
    global process_updates
    while True:
        if process_updates:
            yield f"data: {process_updates.pop(0)}\n\n"
        time.sleep(1)


@app.route('/create_model', methods=['POST'])
def create_model():
    data = request.get_json()
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    if not model_name.endswith(".pkl"):
        return jsonify({"error": "Invalid model type (file is not .pkl)"}), 400
    print(f"Model type: {type(model_name)}")
    r = create_model_pkl(model_name)
    if os.path.exists(r):
        return jsonify({"message": f"Model {model_name} created"}), 200
    else:
        return jsonify({"error": f"Failed to create model: {r}"}), 400


@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    model_path = os.path.join("models", model_name)

    if not file_names:
        return jsonify({"error": "No files provided"}), 400
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    if not model_name.endswith(".pkl"):
        return jsonify({"error": "Invalid model type (file is not .pkl)"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": "Model could not be validated"}), 400

    threading.Thread(target=modelClass.train, args=(file_names, model_name, update_training_status)).start()

    return jsonify({"message": f"Model training started for {model_name}"}), 200


@app.route('/train_model_updates', methods=['GET'])
def train_model_updates():
    return Response(generate_training_output(), mimetype='text/event-stream')


@app.route('/process_files', methods=['POST'])
def process_files():
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    model_path = os.path.join("models", model_name)

    if not file_names:
        return jsonify({"error": "No files provided"}), 400
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    if not model_name.endswith(".pkl"):
        return jsonify({"error": "Invalid model type (file is not .pkl)"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": "Model could not be validated"}), 400

    threading.Thread(target=modelClass.process_pdfs, args=(file_names, model_name, update_process_status)).start()

    return jsonify({"message": "File processing started."}), 200


@app.route('/process_file_updates', methods=['GET'])
def process_file_updates():
    return Response(generate_process_output(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
