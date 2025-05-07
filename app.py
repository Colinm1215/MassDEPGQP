import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from flask_uploads import UploadSet, configure_uploads, UploadNotAllowed
from werkzeug.utils import secure_filename
from pdfScript import get_clean_dataframe
from worker import celery_train_model, celery_process_files
import redis
from database import db, ModelSystem, print_all_db_records
from model import create_model_record
from flask_cors import CORS

# Configuration
app = Flask(__name__)
app.config.from_object('Config.Config')
CORS(app, supports_credentials=True)
files_upload = UploadSet('files', app.config['ALLOWED_EXTENSIONS'])
app.config['UPLOADED_FILES_DEST'] = app.config['UPLOAD_FOLDER']
configure_uploads(app, files_upload)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Database setup
with app.app_context():
    db.init_app(app)
    db.create_all()

def allowed_file(filename):
    """
    Checks if a given filename has an allowed extension based on app configuration.

    Args:
        filename: The name of the file to check.

    Returns:
        bool: True if the file has a valid extension listed in app.config['ALLOWED_EXTENSIONS'], False otherwise.

    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_last_modified(file_path):
    """
    Returns the last modified timestamp of a given file as a formatted string.

    Args:
        file_path: The full path to the file.

    Returns:
        str: The last modified time in 'YYYY-MM-DD HH:MM:SS' format.
    """
    last_modified_timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(last_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')

def event_stream():
    """
    Generator function that listens to Redis Pub/Sub channel 'celery_updates'
    and yields server-sent events (SSE) in real time.
    Subscribes to the 'celery_updates' channel via Redis.
    For each published message of type 'message', yields a formatted SSE string.

    Yields:
        str: Formatted server-sent event line
    """
    pubsub = redis_client.pubsub()
    pubsub.subscribe("celery_updates")
    for message in pubsub.listen():
        if message["type"] == "message":
            yield f"data: {message['data'].decode('utf-8')}\n\n"

@app.route('/task_updates')
def task_updates():
    """
    Endpoint to stream task updates from Redis using Server-Sent Events (SSE).

    Returns:
        - A stream of task updates as Server-Sent Events.
    """
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Endpoint to handle uploading of multiple files via a multipart/form-data request.

    Expects:
        - Form field 'files': A list of files uploaded by the client.

    Returns:
        - JSON response containing a list of per-file upload results
        - HTTP 400 if 'files' field is missing from the request.
    """
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
    """
    Endpoint to render the index page.

    Returns:
        - Rendered HTML template for the index page.
    """
    return render_template('upload.html')

@app.route('/storage', methods=['GET'])
def storage():
    """
    Endpoint to render the storage page.

    Returns:
        - Rendered HTML template for storage page with lists of uploaded files and their last modified dates.
    """
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files_list = []
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        files_list.append({"file": file, "date": get_file_last_modified(file_path)})
    return render_template('storage.html', files=files_list)

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Endpoint to retrieve a list of model names from the database.

    Returns:
        - JSON response containing a list of model names.
    """
    # Query DB for model records.
    model_records = ModelSystem.query.all()
    models_list = [record.name for record in model_records]
    return jsonify(models_list)

@app.route('/get_uploads', methods=['GET'])
def get_uploads():
    """
    Endpoint to retrieve a list of uploaded files in the upload folder.

    Returns:
        - JSON response containing a list of uploaded file paths.
    """
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify([f for f in files if allowed_file(f)])

@app.route('/get_processed_files', methods=['GET'])
def get_processed_files():
    """
    Endpoint to retrieve a list of processed files in the processed folder.

    Returns:
        - JSON response containing a list of processed file paths.
    """
    processed_folder = app.config['PROCESSED_FOLDER']
    processed_files_list = []
    for root, _, files in os.walk(processed_folder):
        for file in files:
            if file.endswith('.csv'):
                relative_path = os.path.relpath(os.path.join(root, file), processed_folder)
                processed_files_list.append(relative_path.replace("\\", "/"))
    return jsonify(processed_files_list)

@app.route('/model', methods=['GET'])
def model():
    """
    Endpoint to render the model page.

    Returns:
        - Rendered HTML template for model page with lists of uploaded CSV files and model names.
    """
    # List uploaded CSV files.
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files_list = []
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            files_list.append({"file": file, "date": get_file_last_modified(file_path)})
    # Get model names from DB.
    model_records = ModelSystem.query.all()
    models_list = [record.name for record in model_records]
    return render_template('model.html', files=files_list, models=models_list)

@app.route('/modelOutputs', methods=['GET'])
def modelOutputs():
    """
    Endpoint to render the model outputs page.

    Returns:
        - Rendered HTML template for model outputs.
    """
    return render_template('modelOutputs.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_files():
    """
    Endpoint to preprocess a list of uploaded files by cleaning their contents
    and saving the results as CSV files in the upload directory.

    Expects a JSON payload with:
        - 'files': A list of file names to preprocess (relative to the 'uploads' directory).

    Returns:
        - HTTP 200 with a success message and list of processed files if all succeed.
        - HTTP 400 if no files are provided in the request.
        - HTTP 500 with an error message if an exception occurs during processing.
    """
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
    """
    Endpoint to delete a single file from either the upload or processed directory.

    Expects form data with:
        - 'file': The path or name of the file to be deleted.

    Returns:
        - HTTP 200 with success message if file is deleted.
        - HTTP 400 if no file name is provided.
        - HTTP 404 if file is not found.
        - HTTP 500 if deletion fails due to an internal error.
    """
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
    """
    Endpoint to delete multiple files from either the upload or processed directories.

    Expects a JSON payload with:
        - 'files': A list of file paths (relative or full) to be deleted.

    For each file:
        - Attempts to securely resolve the path and delete the file.
        - Collects and logs any failures for individual files.

    Returns:
        - HTTP 200 JSON response summarizing success and/or failure
        - HTTP 400 JSON response if no files are provided.
    """
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
    """
    Endpoint to create a new model entry in the database.

    Expects a JSON payload with:
        - 'model_name': The desired name for the new model

    Checks for the presence of a model name and ensures it does not already exist in the database.
    If valid and unique, creates a new model record via `create_model_record`.

    Returns:
        - HTTP 400 with error message if:
            - Model name is missing.
            - Model with the given name already exists.
            - Model creation fails for any reason.
        - HTTP 200 with success message if model creation succeeds.
    """
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
    """
    Validates that a model name is provided and corresponds to an existing model in the database.

    Args:
        model_name (str): The name of the model to validate.

    Returns:
        - A Flask JSON response with an error message and HTTP 400 status if validation fails.
        - None if the model name is valid and the model exists in the database.
    """
    if not model_name:
        return jsonify({"error": "No model name provided"}), 400
    existing = ModelSystem.query.filter_by(name=model_name).first()
    if not existing:
        return jsonify({"error": "Model could not be validated"}), 400
    return None

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Endpoint to initiate asynchronous model training using primary and standardized training files.

    Expects a JSON payload with:
        - 'primary_files': List of paths or identifiers for raw training input files
        - 'secondary_files': List of paths or identifiers for standardized/corrected versions of the primary files
        - 'model_name': Name of the model to be trained.
        - 'bypass_already_trained': Boolean flag to force retraining even if the model was previously trained.

    Validates the presence of required files and the model name.
    If valid, dispatches a Celery task to train the model in the background.

    Returns:
        - HTTP 400 with error message if required files are missing.
        - HTTP 200 with task ID if training starts successfully.
        - Validation error response if the model name is invalid.
    """
    data = request.get_json()
    primary_files = data.get("primary_files", [])
    print(primary_files)
    if not primary_files:
        return jsonify({"error": "No training files provided"}), 400
    secondary_files = data.get("secondary_files", [])
    print(secondary_files)
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
    """
    Endpoint to initiate asynchronous processing of uploaded files using a specified model.

    Expects a JSON payload with:
        - 'files': A list of filenames to process.
        - 'model_name': The name of the model to use for processing.

    Validates the provided model name. If invalid, returns the validation error.
    If valid, triggers a Celery background task to process the files asynchronously.

    Returns:
        JSON response with a success message and the Celery task ID if processing starts,
        or a validation error if the model name is invalid.
    """
    data = request.get_json()
    file_names = data.get('files')
    model_name = data.get('model_name')
    valid = model_validation(model_name)
    if valid:
        return valid
    task = celery_process_files.delay(file_names, model_name)
    return jsonify({"message": "File processing started.", "task_id": task.id}), 200

@app.route('/download', methods=['GET'])
def download_file():
    """
    Endpoint to download a file from either the processed directory.

    Expects:
        - file: Path to the file (relative to the uploads or processed folders)

    Returns:
        - File response if found
        - 400 if parameter is missing
        - 404 if file doesn't exist
    """
    file_path = request.args.get('file')
    if not file_path:
        return jsonify({"error": "No file specified"}), 400


    folder = app.config['PROCESSED_FOLDER'] if os.path.dirname(file_path) else app.config['UPLOAD_FOLDER']
    full_path = os.path.join(folder, file_path)

    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory=folder, path=file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
