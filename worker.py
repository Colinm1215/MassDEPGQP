import os
import sys

import pandas as pd
from celery import current_task
from model import ModelLoadStatus, PDFStandardizer
import redis
from celery_app import celery

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
modelClass = PDFStandardizer()

def save_standardized_reports(standardized_df, model_name):
    processed_folder = os.path.join('processed', model_name)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    fname = standardized_df["filename"].iloc[0]
    file_path = os.path.join(processed_folder, f"{fname}_standardized.csv")
    standardized_df.to_csv(file_path, index=False)

def _send_update(message: str):
    print(f"Updating Celery state: {message}")
    redis_client.publish("celery_updates", message)
    current_task.update_state(state='PROGRESS', meta={'status': message})

@celery.task(bind=True)
def celery_train_model(self, training_files, label_files, model_name, bypass_already_trained, standard_format):
    def callback_fn(msg):
        if not hasattr(self, 'error_reported') or not self.error_reported:
            _send_update(msg)
    try:
        from app import app
        with app.app_context():
            print(f"Training model {model_name} with files: {training_files}")
            print(f"Training model {model_name} with labels: {label_files}")
            pdf_dfs_data_training = []
            for fname in training_files:
                csv_path = os.path.join('uploads', fname)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    pdf_dfs_data_training.append(df)
            pdf_dfs_data_label = []
            for fname in label_files:
                path = os.path.join('uploads', fname)
                if not os.path.exists(path):
                    continue
                if fname.endswith('.csv'):
                    df = pd.read_csv(path)
                if fname.endswith('.xlsx'):
                    df = pd.read_excel(path)

                df['filename'] = os.path.splitext(fname)[0]
                pdf_dfs_data_label.append(df)


            error = modelClass.train(pdf_dfs_data_training,
                                     pdf_dfs_data_label,
                                     model_name,
                                     update_callback=callback_fn,
                                     bypass_already_trained=bypass_already_trained,
                                     standard_format=standard_format)
            if error == ModelLoadStatus.SUCCESS:
                _send_update("Training completed.")
                return {'status': 'SUCCESS', 'message': f"Training completed for {model_name}"}
            elif error == ModelLoadStatus.NOT_FOUND:
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
    def callback_fn(msg):
        if not hasattr(self, 'error_reported') or not self.error_reported:
            _send_update(msg)
    try:
        from app import app
        with app.app_context():
            pdf_dfs_data = []
            for fname in file_names:
                csv_path = os.path.join('uploads', fname)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    pdf_dfs_data.append(df)
            error, standardized_dfs = modelClass.process_pdfs(pdf_dfs_data, model_name, update_callback=callback_fn)
            if error == ModelLoadStatus.SUCCESS:
                _send_update("File processing completed.")
                for df in standardized_dfs:
                    save_standardized_reports(df, model_name)
                return {'status': 'SUCCESS', 'message': "File processing completed."}
            elif error == ModelLoadStatus.UNTRAINED:
                self.error_reported = True
                _send_update("Model must be trained first")
                return {'status': 'ERROR', 'message': "Model must be trained first"}
            elif error == ModelLoadStatus.NOT_FOUND:
                self.error_reported = True
                _send_update("Model not found.")
                return {'status': 'ERROR', 'message': "Model not found."}
            else:
                self.error_reported = True
                _send_update("File processing error")
                return {'status': 'ERROR', 'message': "File processing error"}
    except Exception as e:
        _send_update(f"Unexpected error: {str(e)}")
        return {'status': 'ERROR', 'message': f"Unexpected error: {str(e)}"}
