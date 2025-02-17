import os
import pandas as pd
from celery import Celery, current_task
from model import ModelLoadStatus, PDFStandardizer
import redis

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

celery = Celery(
    'MassDEP',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    result_expires=3600,
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_serializer='pickle',
    timezone='UTC',
    enable_utc=True,
)

celery.conf.update(
    broker_heartbeat=0,
    broker_transport_options={
        'visibility_timeout': 3600,
        'socket_keepalive': True,
        'tcp_keepalive': True,
    }
)

modelClass = PDFStandardizer()

def save_standardized_reports(standardized_df):
    processed_folder = 'processed'
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    fname = standardized_df["filename"].iloc[0]
    file_path = os.path.join(processed_folder, f"{fname}_standardized.csv")
    pd.DataFrame(standardized_df).to_csv(file_path)

def _send_update(message: str):
    print(f"Updating Celery state: {message}")
    redis_client.publish("celery_updates", message)
    current_task.update_state(state='PROGRESS', meta={'status': message})

@celery.task(bind=True)
def celery_train_model(self, file_names, model_name, bypass_already_trained):
    try:
        pdf_dfs_data = []
        for fname in file_names:
            csv_path = os.path.join('uploads', fname)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                pdf_dfs_data.append(df)

        def callback_fn(msg):
            if not hasattr(self, 'error_reported') or not self.error_reported:
                _send_update(msg)

        error = modelClass.train(pdf_dfs_data, model_name, update_callback=callback_fn, bypass_already_trained=bypass_already_trained)

        if error == ModelLoadStatus.SUCCESS:
            _send_update("Training completed.")
            return {'status': 'SUCCESS', 'message': f"Training completed. {modelClass.current_loaded_model}"}
        elif error == ModelLoadStatus.FILE_NOT_FOUND:
            self.error_reported = True
            _send_update("Model not found.")
            return {'status': 'ERROR', 'message': "Model not found."}
        else:
            self.error_reported = True
            _send_update("Training error")
            return {'status': 'ERROR', 'message': "Training error"}
    except Exception as e:
        _send_update(f"Unexpected error: {str(e)}")
        return {'status': 'ERROR', 'message': f"Unexpected error: {str(e)}"}

@celery.task(bind=True)
def celery_process_files(self, file_names, model_name):
    try:
        pdf_dfs_data = []
        for fname in file_names:
            csv_path = os.path.join('uploads', fname)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                pdf_dfs_data.append(df)

        def callback_fn(msg):
            if not hasattr(self, 'error_reported') or not self.error_reported:
                _send_update(msg)

        error, standardized_dfs = modelClass.process_pdfs(pdf_dfs_data, model_name, update_callback=callback_fn)

        if error == ModelLoadStatus.SUCCESS:
            _send_update("File processing completed.")
            for df in standardized_dfs:
                save_standardized_reports(df)
            return {'status': 'SUCCESS', 'message': "File processing completed."}
        elif error == ModelLoadStatus.UNTRAINED:
            self.error_reported = True
            _send_update("Model must be trained first")
            return {'status': 'ERROR', 'message': "Model must be trained first"}
        elif error == ModelLoadStatus.FILE_NOT_FOUND:
            self.error_reported = True
            _send_update("File not found.")
            return {'status': 'ERROR', 'message': "File not found."}
        else:
            self.error_reported = True
            _send_update("File processing error")
            return {'status': 'ERROR', 'message': "File processing error"}
    except Exception as e:
        _send_update(f"Unexpected error: {str(e)}")
        return {'status': 'ERROR', 'message': f"Unexpected error: {str(e)}"}