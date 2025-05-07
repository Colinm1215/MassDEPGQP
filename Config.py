import os

class Config:
    """
    Centralized Flask application configuration.
    """
    SQLALCHEMY_DATABASE_URI = "sqlite:///models.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')
    MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
    ALLOWED_EXTENSIONS = {'pdf', 'csv'}
