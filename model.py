import gc
import json
import os
from datetime import datetime

import spacy
from spacy.pipeline import EntityRuler
from enum import Enum
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer, \
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from werkzeug.utils import secure_filename
from database import db, ModelSystem, ModelStatus, LLMModel, TrainStatus, NamedEntity, TrainingData
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from pdfScript import get_clean_dataframe


# ModelLoadStatus defines possible outcomes for model loading/training
class ModelLoadStatus(Enum):
    SUCCESS = 1
    NOT_FOUND = 2
    UNTRAINED = 3
    ERROR = 4

def create_model_record(model_name):
    """
    Create a new model record in the database if it doesn't already exist.

    Args:
        model_name: Unique model name

    Returns:
        ModelSystem: SQLAlchemy model instance
    """
    existing = ModelSystem.query.filter_by(name=model_name).first()
    if existing:
        return existing
    new_model = ModelSystem(name=model_name, status=ModelStatus.UNTRAINED.value)
    db.session.add(new_model)
    db.session.commit()
    return new_model

class PDFStandardizer:
    """
    Orchestrates PDF preprocessing, LLM fine-tuning, entity extraction,
    standardization, validation, and storage.

    Components:
        - T5 generator for standardization
        - RoBERTa discriminator for validation
        - spaCy NLP pipeline with custom EntityRuler
        - SQLAlchemy for metadata persistence

    All models are stored locally under `models/` and all training progress
    is tracked in the database.
    """

    def __init__(self):
        """
        Initialize NLP pipeline, load pre-trained LLMs, define EntityRuler patterns,
        and set up device (CPU/GPU). If `models/` folder doesn't exist, create it.
        """

        if not os.path.exists("models"):
            os.makedirs("models")
        self.nlp = spacy.load("en_core_web_trf")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.discriminator = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        self.discriminator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_model.to(self.device)
        self.discriminator.to(self.device)
        self.standard_format = ""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = [
            {"label": "ORDER_NUMBER", "pattern": [
                {"LOWER": "order"}, {"IS_PUNCT": True, "OP": "*"},
                {"TEXT": {"REGEX": r"[#:]?"}, "OP": "?"},
                {"TEXT": {"REGEX": r"[A-Za-z0-9\-]+"}}
            ]},

            {"label": "DATE", "pattern": [
                {"TEXT": {"REGEX": r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"}}
            ]},

            {"label": "SOURCE_LOCATION", "pattern": [
                {"LOWER": "source"}, {"LOWER": "location"},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]},

            {"label": "DESTINATION", "pattern": [
                {"LOWER": "destination"},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]},

            {"label": "ADDRESS", "pattern": [
                {"LIKE_NUM": True},
                {"IS_ALPHA": True, "OP": "+"},
                {"LOWER": {"IN": ["ave", "avenue", "st", "street", "road", "rd", "blvd", "drive", "dr", "lane", "ln"]},
                 "OP": "?"}
            ]},

            {"label": "SOIL_TYPE", "pattern": [
                {"LOWER": "soil"}, {"LOWER": "type"},
                {"IS_PUNCT": True, "OP": "?"},
                {"LOWER": {"IN": ["sandy", "clay", "loam", "silt", "gravel"]}, "OP": "+"}
            ]},

            {"label": "VOLUME_CUYD", "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["cubic", "cu", "cuyds", "yds"]}},
                {"LOWER": {"IN": ["yard", "yards", "yd"]}, "OP": "?"}
            ]},

            {"label": "WEIGHT_TON", "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["ton", "tons", "tonnage", "net", "gross", "tare"]}}
            ]},

            {"label": "LOAD_COUNT", "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["load", "loads"]}}
            ]},

            {"label": "SAFETY_STATUS", "pattern": [
                {"LOWER": "safety"}, {"LOWER": "status"},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]},

            {"label": "SPECIAL_HANDLING", "pattern": [
                {"LOWER": "special"}, {"LOWER": "handling"},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]},

            {"label": "TEST_RESULT", "pattern": [
                {"LOWER": "test"},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]},

            {"label": "PHONE_NUMBER", "pattern": [
                {"TEXT": {"REGEX": r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"}}
            ]},

            {"label": "EXCEEDANCE", "pattern": [
                {"LOWER": {"IN": ["exceed", "exceeds", "exceedance", "exceedances"]}}
            ]},

            {"label": "ABOVE_LIMIT", "pattern": [
                {"LOWER": "above"},
                {"LOWER": {"IN": ["criteria", "limit", "threshold", "rcs-1", "rcs-2"]}, "OP": "?"}
            ]},

            {"label": "RCS_LEVEL", "pattern": [
                {"TEXT": {"REGEX": r"RCS[- ]?[12]"}}
            ]},

            {"label": "ACCEPTANCE_CRITERIA", "pattern": [
                {"LOWER": "acceptance"}, {"LOWER": "criteria"}
            ]},

            {"label": "REJECTION", "pattern": [
                {"LOWER": {"IN": ["rejected", "rejection", "quarantined", "quarantine"]}}
            ]},

            {"label": "THIRD_PARTY_INSPECTION", "pattern": [
                {"LOWER": {"IN": ["third", "3rd"]}},
                {"LOWER": {"IN": ["party", "inspector", "inspection", "tpi"]}}
            ]},

            {"label": "INSPECTION_DATE", "pattern": [
                {"LOWER": {"IN": ["inspection", "inspected"]}},
                {"LOWER": {"IN": ["on", "at"]}, "OP": "?"},
                {"TEXT": {"REGEX": r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"}}
            ]},

            {"label": "NEGATION", "pattern": [
                {"LOWER": {"IN": ["no", "not", "none", "without"]}}
            ]}
        ]

        ruler.add_patterns(patterns)

    def fine_tune(self, model_name, training_data, label_data, standard_format, entity_data=None,
                  epochs=3, batch_size=4, lr=5e-5, update_callback=None):
        """
        Fine-tune a FLAN-T5 model on a custom standardization task.
        Adds discriminator feedback to ensure factual accuracy and format coherence.

        Args:
            model_name: Target model ID.
            training_data: Raw extracted PDF text.
            label_data: Expected standardized outputs.
            standard_format: Template format for structured output.
            entity_data: Named entities per training sample.
            epochs: Training loop count.
            batch_size: Samples per gradient step.
            lr: Learning rate.
            update_callback: Real-time logger.

        Returns:
            dict: Training summary including loss, config, paths, and entity metadata.
        """

        if update_callback:
            update_callback(
                f"[DEBUG] Starting fine_tune for model '{model_name}' with epochs={epochs}, batch_size={batch_size}, lr={lr}")
        optimizer = Adam(self.t5_model.parameters(), lr=lr)
        ce_loss = CrossEntropyLoss(ignore_index=self.t5_tokenizer.pad_token_id)
        bce_loss = BCEWithLogitsLoss()

        dataset_size = len(training_data)
        if update_callback:
            update_callback(f"[DEBUG] Dataset size: {dataset_size}")
        self.t5_model.train()
        self.discriminator.eval()

        loss_history = []
        last_gen_loss = None
        last_disc_loss = None

        for epoch in range(epochs):
            if update_callback:
                update_callback(f"[DEBUG] Epoch {epoch + 1}/{epochs} started.")
            epoch_losses = []

            for i in range(0, dataset_size, batch_size):
                batch_num = i // batch_size + 1
                start_idx = i
                end_idx = min(i + batch_size, dataset_size)
                if update_callback:
                    update_callback(f"[DEBUG] Processing batch {batch_num}: samples {start_idx+1} to {end_idx+1}")

                # Prepare prompts and labels
                batch_inputs, batch_labels = [], []
                for j in range(start_idx, end_idx):
                    entity_info = f"Entities: {entity_data[j]}" if entity_data else ""
                    prompt = (
                        f"""
                        You are a data standardization expert. Your task is to standardize the following text:
                        {training_data[j]}
                        standardize the text according to the following format:
                        {standard_format}
                        and make sure that the following named entities and any other relevant factual information are included:
                        {entity_info}
                        """
                    )
                    batch_inputs.append(prompt)
                    batch_labels.append(label_data[j])

                # Tokenization
                if update_callback:
                    update_callback(f"[DEBUG] Tokenizing batch {batch_num}")
                input_encodings = self.t5_tokenizer(batch_inputs, return_tensors="pt",
                                                    padding=True, truncation=True, max_length=256)
                label_encodings = self.t5_tokenizer(batch_labels, return_tensors="pt",
                                                    padding=True, truncation=True, max_length=256)

                input_ids = input_encodings.input_ids.to(self.device)
                attention_mask = input_encodings.attention_mask.to(self.device)
                labels = label_encodings.input_ids.to(self.device)

                # Generator forward pass
                outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                gen_loss = outputs.loss
                last_gen_loss = gen_loss.item()
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} gen_loss: {last_gen_loss:.4f}")

                # Generation step
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} starting generate() call")
                with torch.no_grad():
                    generated_ids = self.t5_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=64,
                        num_beams=2,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} generate() completed")

                generated_texts = [self.t5_tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                sample_text = generated_texts[0] if generated_texts else 'N/A'
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} sample generated text: {sample_text}")

                # Discriminator input
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} preparing discriminator inputs")
                disc_inputs = self.discriminator_tokenizer(
                    training_data[start_idx:end_idx],
                    generated_texts,
                    return_tensors="pt",
                    padding=True, truncation=True, max_length=512
                )
                disc_inputs = {k: v.to(self.device) for k, v in disc_inputs.items()}
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} discriminator tokenization completed")

                with torch.no_grad():
                    disc_outputs = self.discriminator(**disc_inputs)
                    target = torch.ones(disc_outputs.logits.shape, device=self.device)
                    disc_loss = bce_loss(disc_outputs.logits, target)

                last_disc_loss = disc_loss.item()
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} disc_loss: {last_disc_loss:.4f}")

                # Total loss and backward
                total_loss = gen_loss + 0.5 * disc_loss
                if update_callback:
                    update_callback(f"[DEBUG] Batch {batch_num} total_loss: {total_loss.item():.4f}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())
                if update_callback:
                    update_callback(f"Batch {batch_num}: gen_loss={last_gen_loss:.4f}, disc_loss={last_disc_loss:.4f}")

                torch.cuda.empty_cache()
                gen_loss = outputs.loss.detach()
                disc_loss = disc_loss.detach()
                del outputs, generated_ids, input_ids, attention_mask, labels
                del disc_inputs, disc_outputs

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            loss_history.append({"epoch": epoch + 1, "average_loss": avg_loss, "batch_losses": epoch_losses})
            if update_callback:
                update_callback(f"[DEBUG] Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        # Save models
        generator_weights_path = f"models/{model_name}_finetuned_generator.pt"
        discriminator_weights_path = f"models/{model_name}_finetuned_discriminator.pt"
        self.t5_model.save_pretrained(generator_weights_path)
        self.discriminator.save_pretrained(discriminator_weights_path)

        if update_callback:
            update_callback(
                f"[DEBUG] Fine-tuning completed. Final gen_loss: {last_gen_loss:.4f}, disc_loss: {last_disc_loss:.4f}")
            update_callback("Fine-tuning completed.")
        return {
            "final_loss": total_loss.item(),
            "generator_loss": last_gen_loss,
            "discriminator_loss": last_disc_loss,
            "epochs": epochs,
            "loss_history": loss_history,
            "hyperparameters": {"learning_rate": lr, "batch_size": batch_size, "epochs": epochs},
            "training_status": "trained",
            "generator_weights_path": generator_weights_path,
            "discriminator_weights_path": discriminator_weights_path,
            "training_data": training_data,
            "named_entities": entity_data or [],
            "standard_format": standard_format
        }

    def extract_entities(self, text, update_callback):
        """
        Extract entities from raw text using the spaCy pipeline with custom rules.

        Args:
            text: Unstructured input text
            update_callback: Real-time logger.

        Returns:
            List[dict]: Extracted entity_type and entity_value pairs
        """
        import re

        doc = self.nlp(text)
        raw_ents = list(doc.ents)
        results = []
        skip_indices = set()

        for i, ent in enumerate(raw_ents):
            if i in skip_indices:
                continue

            label = ent.label_
            value = ent.text.strip()

            # --- 1. Negation: "no exceedances" → "NO_EXCEEDANCE"
            if label == "EXCEEDANCE":
                for j in range(max(0, ent.start - 3), ent.start):
                    if doc[j].ent_type_ == "NEGATION":
                        label = "NO_EXCEEDANCE"
                        break

            # --- 2. Normalize volume (cu yd or tons)
            elif label in {"VOLUME_CUYD", "WEIGHT_TON"}:
                label = "SOIL_VOLUME"
                value = f"{value} ({ent.label_})"

            # --- 3. Merge adjacent GPEs like "Springfield, MA"
            elif label == "GPE":
                if i + 1 < len(raw_ents):
                    next_ent = raw_ents[i + 1]
                    if next_ent.label_ == "GPE" and ent.end == next_ent.start:
                        value = f"{value}, {next_ent.text.strip()}"
                        label = "SOURCE_LOCATION_GPE"
                        skip_indices.add(i + 1)

            # --- 4. Trim prefixes for order numbers and destinations
            elif label == "ORDER_NUMBER":
                value = re.sub(r"^Order\s*[:#-]*\s*", "", value, flags=re.IGNORECASE)

            elif label == "DESTINATION":
                value = re.sub(r"^Destination\s*[:]*\s*", "", value, flags=re.IGNORECASE)

            elif label == "SOURCE_LOCATION":
                value = re.sub(r"^Source\s+Location\s*[:]*\s*", "", value, flags=re.IGNORECASE)

            # --- 5. Split out concatenated entities (e.g., "30 cubic yards Safety Status")
            if label == "ADDRESS" and "safety status" in value.lower():
                parts = re.split(r"\s+(?=Safety Status)", value, maxsplit=1)
                if len(parts) == 2:
                    results.append({"entity_type": "ADDRESS", "entity_value": parts[0].strip()})
                    results.append({"entity_type": "SAFETY_STATUS", "entity_value": parts[1].strip()})
                    continue

            # --- 6. Fix overmatch in SPECIAL_HANDLING
            if label == "SPECIAL_HANDLING" and "Requested Delivery Date" in value:
                parts = re.split(r"Requested Delivery Date", value, maxsplit=1)
                results.append({"entity_type": label, "entity_value": parts[0].strip()})
                continue

            # --- 7. Fix truncated or misclassified phone numbers
            if label in {"CARDINAL", "OTHER"}:
                match = re.search(r"\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}", value)
                if match:
                    label = "PHONE_NUMBER"
                    value = match.group(0)

            results.append({
                "entity_type": label,
                "entity_value": value
            })

        if update_callback:
            update_callback(f"Extracted {len(results)} entities: {results}")

        return results

    def generate_standardized_report(self, text, update_callback=None):
        """
         Generate a standardized version of a text input using T5.

         Args:
             text: Raw input text
             update_callback: Real-time logger

         Returns:
             str: Standardized output
         """
        entities = self.extract_entities(text, update_callback=update_callback)
        entity_info = f"Entities: {json.dumps(entities)}" if entities else ""

        prompt = (
            f"""
            You are a data standardization expert. Your task is to standardize the following text:
            {text}
            standardize the text according to the following format:
            {self.standard_format}
            and make sure that the following named entities and any other relevant factual information are included:
            {entity_info}
            """
        )

        input_encodings = self.t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = input_encodings.input_ids.to(self.device)
        attention_mask = input_encodings.attention_mask.to(self.device)

        generated_ids = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        return self.t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def validate_report(self, original_text, generated_text):
        """
        Use discriminator to assess semantic consistency between original and generated text.

        Args:
            original_text: Original unstructured text
            generated_text: Generated standardized text

        Returns:
            torch.Tensor: Probability of semantic entailment (confidence score)
        """
        from torch.nn.functional import softmax
        inputs = self.discriminator_tokenizer(original_text, generated_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.discriminator(**inputs)
        return softmax(outputs.logits, dim=-1)[:, 2]

    def load_model(self, model_name, update_callback=None):
        """
        Load model weights from disk and restore standard_format.

        Args:
            model_name: Unique model name
            update_callback: Real-time logger

        Returns:
            ModelLoadStatus: Enum indicating outcome
        """
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            return ModelLoadStatus.NOT_FOUND
        if record.status == ModelStatus.UNTRAINED.value:
            return ModelLoadStatus.UNTRAINED

        self.standard_format = record.standard_format

        generator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="generator").first()
        discriminator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="discriminator").first()

        try:
            if generator_llm and generator_llm.weights_path and os.path.exists(generator_llm.weights_path):
                del self.t5_model
                gc.collect()
                torch.cuda.empty_cache()
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(generator_llm.weights_path)
                self.t5_model.to(self.device)
            else:
                if update_callback:
                    update_callback(f"[ERROR] could not load weights for {model_name}")
                return ModelLoadStatus.ERROR

            if discriminator_llm and discriminator_llm.weights_path and os.path.exists(discriminator_llm.weights_path):
                del self.discriminator
                gc.collect()
                torch.cuda.empty_cache()
                self.discriminator = AutoModelForSequenceClassification.from_pretrained(discriminator_llm.weights_path)
                self.discriminator.to(self.device)
            else:
                if update_callback:
                    update_callback(f"[ERROR] could not load weights for {model_name}")
                return ModelLoadStatus.ERROR
        except Exception as e:
            if update_callback:
                update_callback(f"[ERROR] : {e}")
            return ModelLoadStatus.ERROR

        return ModelLoadStatus.SUCCESS

    def train(self, pdf_dfs_training, pdf_dfs_label, model_name, update_callback=None, bypass_already_trained=False, standard_format=""):
        """
        Train both generator and discriminator using cleaned PDF dataframes.

        Args:
            pdf_dfs_training: Raw text by page
            pdf_dfs_label: Clean labels by page
            model_name: Name of the model
            update_callback: Optional logger
            bypass_already_trained: Force retrain
            standard_format: Template used for generation

        Returns:
            ModelLoadStatus: Enum indicating outcome
        """
        status = self.load_model(model_name)
        if status == ModelLoadStatus.SUCCESS and not bypass_already_trained:
            if update_callback:
                update_callback("Model is already trained, skipping training.")
            return ModelLoadStatus.SUCCESS
        if update_callback:
            update_callback(f"Training started for {model_name}...")

        # Prepare training data and entity information
        training_data = []
        entity_data = []
        for df_index, df in enumerate(pdf_dfs_training):
            for text in df["Text"]:
                training_data.append(text)
                entity_data.append(self.extract_entities(text, update_callback))

        label_data = []
        for df_index, df in enumerate(pdf_dfs_label):
            for text in df["Text"]:
                label_data.append(text)

        fine_tune_result = self.fine_tune(
            model_name=model_name,
            training_data=training_data,
            label_data=label_data,
            standard_format=standard_format,
            entity_data=entity_data,
            update_callback=update_callback
        )

        if update_callback:
            update_callback(f"Training info for {model_name}: {fine_tune_result}")

        # Update the model record in the DB to mark it as TRAINED.
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            record = create_model_record(model_name)
        record.status = ModelStatus.TRAINED.value

        # Save the training configuration and history as JSON strings.
        record.train_config = json.dumps(fine_tune_result["hyperparameters"])
        record.train_history = json.dumps(fine_tune_result["loss_history"])
        record.standard_format = fine_tune_result["standard_format"]

        # Update or create the generator LLM record.
        generator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="generator").first()
        if not generator_llm:
            generator_llm = LLMModel(model_system_id=record.id, llm_type="generator")
        generator_llm.architecture = "t5-small"
        generator_llm.weights_path = fine_tune_result["generator_weights_path"]
        generator_llm.hyperparameters = json.dumps(fine_tune_result["hyperparameters"])
        generator_llm.train_status = TrainStatus.TRAINED.value
        db.session.add(generator_llm)

        # Update or create the discriminator LLM record.
        discriminator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="discriminator").first()
        if not discriminator_llm:
            discriminator_llm = LLMModel(model_system_id=record.id, llm_type="discriminator")
        discriminator_llm.architecture = "roberta-base"
        discriminator_llm.weights_path = fine_tune_result["discriminator_weights_path"]
        discriminator_llm.hyperparameters = json.dumps(fine_tune_result["hyperparameters"])
        discriminator_llm.train_status = TrainStatus.TRAINED.value
        db.session.add(discriminator_llm)

        # Update Named Entities.
        nested = fine_tune_result.get("named_entities", [])
        flat_entities = []
        for item in nested:
            if isinstance(item, dict):
                flat_entities.append(item)
                update_callback(f"Extracted entity as list somehow: {item}")
            elif isinstance(item, list):
                flat_entities.extend([e for e in item if isinstance(e, dict)])

        for entity in flat_entities:
            new_entity = NamedEntity(
                model_system_id=record.id,
                entity_type=entity.get("entity_type"),
                entity_value=entity.get("entity_value")
            )
            db.session.add(new_entity)

        # Update the TrainingData table
        for file in pdf_dfs_training:
            file_name = str(file["filename"].iloc[0]) if "filename" in file.columns else "unknown"
            file_path = f"uploads/{file_name}"
            processed_at = datetime.utcnow()

            if update_callback:
                update_callback(f"""
                Training Data to Save :
                model_system_id={record.id},
                file_name={file_name},
                file_path={file_path},
                processed_at={processed_at}
                """)

            new_td = TrainingData(
                model_system_id=record.id,
                file_name=file_name,
                file_path=file_path,
                processed_at=processed_at
            )
            db.session.add(new_td)

        for file in pdf_dfs_label:
            file_name = str(file["filename"].iloc[0]) if "filename" in file.columns else "unknown"
            file_path = f"uploads/{file_name}"
            processed_at = datetime.utcnow()

            if update_callback:
                update_callback(f"""
                Label Data to Save :
                model_system_id={record.id},
                file_name={file_name},
                file_path={file_path},
                processed_at={processed_at}
                """)

            new_td = TrainingData(
                model_system_id=record.id,
                file_name=file_name,
                file_path=file_path,
                processed_at=processed_at
            )
            db.session.add(new_td)

        db.session.commit()

        if update_callback:
            update_callback(f"Training completed. Model {model_name} is now TRAINED.")
        return ModelLoadStatus.SUCCESS

    def process_pdfs(self, pdf_dfs, model_name, update_callback=None, num_tries=3):
        """
        Use a trained model to process raw PDF dataframe into standardized outputs.

        Args:
            pdf_dfs: List of files (one DataFrame per file).
            model_name: Which trained model to use.
            update_callback: Optional log function.
            num_tries: Max retry attempts if low confidence.

        Returns:
            Tuple[ModelLoadStatus, List[pd.DataFrame]]: Status and output DataFrames
        """
        status = self.load_model(model_name)
        if status != ModelLoadStatus.SUCCESS:
            if update_callback:
                update_callback("Model must be trained first.")
            return status, None
        if update_callback:
            update_callback(f"Processing started for files using {model_name}...")

        reports = []
        for df_index, df in enumerate(pdf_dfs):
            standardized_report = []
            if update_callback:
                update_callback(f"Processing file {df_index+1} of {len(pdf_dfs)}...")

            for text_index, text in enumerate(df["Text"]):
                cur_try = 0
                standardized_text = ""
                validated_score = 0

                while validated_score < 0.5:
                    if update_callback:
                        update_callback(f"Processing page {text_index+1}, attempt {cur_try+1}")
                    standardized_text = self.generate_standardized_report(text, update_callback)
                    validated_score = self.validate_report(text, standardized_text).item()
                    if update_callback:
                        update_callback(f"Page {text_index+1} processed with confidence {validated_score}")
                    if cur_try == num_tries-1:
                        if update_callback:
                            update_callback("Maximum number of generation attempts reach, continuing...")
                        break
                    cur_try += 1

                standardized_report.append({
                    "filename": df["filename"].iloc[0],
                    "page_number": df["Page"].iloc[text_index],
                    "original_text": text,
                    "standardized_text": standardized_text,
                    "confidence": validated_score
                })
            new_report_df = pd.DataFrame(standardized_report)
            reports.append(new_report_df)
            if update_callback:
                update_callback(f"Finished Processing file {df_index+1} of {len(pdf_dfs)}")

        output_dir = f"processed/{model_name}"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for report in reports:
            filename = report["filename"].iloc[0]
            report.to_csv(f'{output_dir}/{filename}_Standardized.csv', index=False)

        if update_callback:
            update_callback("Processing completed. Saving data...")
        return ModelLoadStatus.SUCCESS, reports
