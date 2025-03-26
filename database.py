# database.py
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from enum import Enum

db = SQLAlchemy()

class ModelStatus(Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"

class TrainStatus(Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


# 1. Models Table
class ModelSystem(db.Model):
    __tablename__ = 'models'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), unique=True, nullable=False)
    status = db.Column(db.String(50), default=ModelStatus.UNTRAINED.value)  # or an enum
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    train_config = db.Column(db.Text, nullable=True)  # store JSON as text
    train_history = db.Column(db.Text, nullable=True) # store JSON as text

    # Relationship: one ModelSystem -> many LLMModels
    llms = db.relationship("LLMModel", back_populates="model_system", cascade="all, delete-orphan")
    # Relationship: one ModelSystem -> many NamedEntities
    named_entities = db.relationship("NamedEntity", back_populates="model_system", cascade="all, delete-orphan")
    # Relationship: one ModelSystem -> many TrainingData
    training_data = db.relationship("TrainingData", back_populates="model_system", cascade="all, delete-orphan")


# 2. LLM Table
class LLMModel(db.Model):
    __tablename__ = 'llm_models'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_system_id = db.Column(db.String(36), db.ForeignKey('models.id'))
    llm_type = db.Column(db.String(50))  # "generator" or "discriminator"
    architecture = db.Column(db.String(100))  # e.g., "t5-small", "roberta-base"
    weights_path = db.Column(db.String(255))  # points to local disk or S3
    hyperparameters = db.Column(db.Text, nullable=True) # store JSON as text
    train_status = db.Column(db.String(50), default=TrainStatus.UNTRAINED.value)

    model_system = db.relationship("ModelSystem", back_populates="llms")


# 3. Named Entities Table
class NamedEntity(db.Model):
    __tablename__ = 'named_entities'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_system_id = db.Column(db.String(36), db.ForeignKey('models.id'))
    entity_type = db.Column(db.String(50))  # "Location", "Date", ...
    entity_value = db.Column(db.Text)       # actual extracted text

    model_system = db.relationship("ModelSystem", back_populates="named_entities")


# 4. Training Data Table
class TrainingData(db.Model):
    __tablename__ = 'training_data'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_system_id = db.Column(db.String(36), db.ForeignKey('models.id'))
    file_name = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    processed_at = db.Column(db.DateTime)

    model_system = db.relationship("ModelSystem", back_populates="training_data")
