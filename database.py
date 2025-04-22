import uuid
from datetime import datetime
from enum import Enum

from flask_sqlalchemy import SQLAlchemy

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
    status = db.Column(db.String(50), default=ModelStatus.UNTRAINED.value)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    train_config = db.Column(db.Text, nullable=True)  # store JSON as text
    train_history = db.Column(db.Text, nullable=True) # store JSON as text
    standard_format = db.Column(db.Text, nullable=True)

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
    architecture = db.Column(db.String(100))
    weights_path = db.Column(db.String(255))  # points to local disk
    hyperparameters = db.Column(db.Text, nullable=True) # store JSON as text
    train_status = db.Column(db.String(50), default=TrainStatus.UNTRAINED.value)

    model_system = db.relationship("ModelSystem", back_populates="llms")


# 3. Named Entities Table
class NamedEntity(db.Model):
    __tablename__ = 'named_entities'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_system_id = db.Column(db.String(36), db.ForeignKey('models.id'))
    entity_type = db.Column(db.String(50))
    entity_value = db.Column(db.Text)

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

def print_all_db_records():
    print("=== Models ===")
    models = ModelSystem.query.all()
    for m in models:
        print(f"ID: {m.id}, Name: {m.name}, Status: {m.status}, Created: {m.created_at}, Updated: {m.updated_at}")
        print(f"  Train Config: {m.train_config}")
        print(f"  Train History: {m.train_history}")
        print(f"  LLMs: {len(m.llms)}")
        print(f"  Named Entities: {len(m.named_entities)}")
        print(f"  Training Data: {len(m.training_data)}")
        print("")

    print("=== LLM Models ===")
    llms = LLMModel.query.all()
    for llm in llms:
        print(f"ID: {llm.id}, Model System ID: {llm.model_system_id}, Type: {llm.llm_type}, Architecture: {llm.architecture}")
        print(f"  Weights Path: {llm.weights_path}")
        print(f"  Hyperparameters: {llm.hyperparameters}")
        print(f"  Train Status: {llm.train_status}")
        print("")

    print("=== Named Entities ===")
    entities = NamedEntity.query.all()
    for e in entities:
        print(f"ID: {e.id}, Model System ID: {e.model_system_id}, Type: {e.entity_type}")
        print(f"  Value: {e.entity_value}")
        print("")

    print("=== Training Data ===")
    training_data_list = TrainingData.query.all()
    for td in training_data_list:
        print(f"ID: {td.id}, Model System ID: {td.model_system_id}, File Name: {td.file_name}")
        print(f"  File Path: {td.file_path}")
        print(f"  Processed At: {td.processed_at}")
        print("")
