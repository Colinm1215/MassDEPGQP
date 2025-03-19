import json
import os
from datetime import datetime

import spacy
from enum import Enum
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
import pandas as pd
from werkzeug.utils import secure_filename
from database import db, ModelSystem, ModelStatus, LLMModel, TrainStatus, NamedEntity, TrainingData
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


# ModelLoadStatus defines possible outcomes for model loading/training
class ModelLoadStatus(Enum):
    SUCCESS = 1
    NOT_FOUND = 2
    UNTRAINED = 3
    ERROR = 4

def create_model_record(model_name):
    # Create a new model record in the DB if it doesn't already exist.
    existing = ModelSystem.query.filter_by(name=model_name).first()
    if existing:
        return existing
    new_model = ModelSystem(name=model_name, status=ModelStatus.UNTRAINED.value)
    db.session.add(new_model)
    db.session.commit()
    return new_model


class PDFStandardizer:
    def __init__(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        self.nlp = spacy.load("en_core_web_sm")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.discriminator = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.discriminator_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # Place models on device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_model.to(self.device)
        self.discriminator.to(self.device)

    def fine_tune(self, model_name, training_data, label_data, standard_format, entity_data=None,
                  epochs=3, batch_size=4, lr=5e-5, update_callback=None):
        """
        Fine tune the T5 model using training_data and label_data.
        Optionally, entity_data can be concatenated to the input for context.
        The blank_format is used as a prompt template.
        """
        optimizer = Adam(self.t5_model.parameters(), lr=lr)
        ce_loss = CrossEntropyLoss(ignore_index=self.t5_tokenizer.pad_token_id)
        bce_loss = BCEWithLogitsLoss()

        # Assume training_data and label_data are lists of strings of equal length
        dataset_size = len(training_data)
        self.t5_model.train()
        self.discriminator.train()

        loss_history = []
        last_gen_loss = None
        last_disc_loss = None

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            if update_callback:
                update_callback(f"Epoch {epoch + 1}/{epochs} started.")
            for i in range(0, dataset_size, batch_size):
                batch_inputs = []
                batch_labels = []
                for j in range(i, min(i + batch_size, dataset_size)):
                    # Optionally include entity data in the prompt
                    entity_info = ""
                    if entity_data:
                        entity_info = " | Entities: " + str(entity_data[j])
                    # Construct the input prompt
                    prompt = (
                        f"Output Format: {standard_format} "
                        f"and Named Entities: {entity_info} "
                        f"with Input: {training_data[j]}. "
                        "If any factual information is missing, output '{CANNOT FIND}' in its place.")
                    batch_inputs.append(prompt)
                    batch_labels.append(label_data[j])

                # Tokenize inputs and labels
                input_encodings = self.t5_tokenizer(batch_inputs, return_tensors="pt",
                                                    padding=True, truncation=True, max_length=512)
                label_encodings = self.t5_tokenizer(batch_labels, return_tensors="pt",
                                                    padding=True, truncation=True, max_length=512)
                input_ids = input_encodings.input_ids.to(self.device)
                attention_mask = input_encodings.attention_mask.to(self.device)
                labels = label_encodings.input_ids.to(self.device)

                # Forward pass through T5
                outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                gen_loss = outputs.loss  # CrossEntropyLoss computed internally

                # Generate output for discriminator loss computation
                with torch.no_grad():
                    generated_ids = self.t5_model.generate(input_ids=input_ids,
                                                           attention_mask=attention_mask,
                                                           max_length=512)
                generated_texts = [self.t5_tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

                # Use discriminator to check consistency
                disc_inputs = self.discriminator_tokenizer(training_data[i:i + batch_size],
                                                           generated_texts,
                                                           return_tensors="pt",
                                                           padding=True, truncation=True, max_length=512)
                disc_inputs = {k: v.to(self.device) for k, v in disc_inputs.items()}
                disc_outputs = self.discriminator(**disc_inputs)
                # Assume target consistency score is 1 (perfect match) for all examples
                target = torch.ones(disc_outputs.logits.shape, device=self.device)
                disc_loss = bce_loss(disc_outputs.logits, target)

                # Total loss: weighted sum of generation loss and discriminator loss
                total_loss = gen_loss + 0.5 * disc_loss  # weight can be tuned

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())
                last_gen_loss = gen_loss.item()
                last_disc_loss = disc_loss.item()

                if update_callback:
                    update_callback(
                        f"Batch {i // batch_size + 1}: gen_loss={last_gen_loss:.4f}, disc_loss={last_disc_loss:.4f}")

            loss_history.append({
                "epoch": epoch + 1,
                "average_loss": sum(epoch_losses) / len(epoch_losses),
                "batch_losses": epoch_losses
            })

        generator_weights_path = f"models/{model_name}_finetuned.pt"
        discriminator_weights_path = f"models/{model_name}_finetuned.pt"
        self.t5_model.save_pretrained(generator_weights_path)
        self.discriminator.save_pretrained(discriminator_weights_path)

        # After training, update the model record status if using DB (omitted here for brevity)
        if update_callback:
            update_callback("Fine-tuning completed.")
        return {
            "final_loss": total_loss.item(),
            "generator_loss": last_gen_loss,
            "discriminator_loss": last_disc_loss,
            "epochs": epochs,
            "loss_history": loss_history,
            "hyperparameters": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs
            },
            "training_status": "trained",
            "generator_weights_path": generator_weights_path,
            "discriminator_weights_path": discriminator_weights_path,
            "training_data": training_data,
            "named_entities": entity_data if entity_data else []
        }

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {"Locations": [], "Dates": [], "Quantities": [], "Other": []}
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                entities["Locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["Dates"].append(ent.text)
            elif ent.label_ in ["CARDINAL", "QUANTITY"]:
                entities["Quantities"].append(ent.text)
            else:
                entities["Other"].append(ent.text)
        return entities

    def generate_standardized_report(self, text, max_input_length=512, chunk_overlap=0):
        prefix = "standardize: "
        prefix_ids = self.t5_tokenizer.encode(prefix, add_special_tokens=False)
        text_ids = self.t5_tokenizer.encode(text, add_special_tokens=False)
        available_length = max_input_length - len(prefix_ids)
        if available_length <= 0:
            raise ValueError("max_input_length is too small to accommodate the prefix.")
        chunks = []
        start = 0
        while start < len(text_ids):
            end = start + available_length
            chunks.append(text_ids[start:end])
            start = end - chunk_overlap if end < len(text_ids) else end
        output_chunks = []
        for chunk in chunks:
            combined_ids = prefix_ids + chunk
            input_ids = self.t5_tokenizer(
                self.t5_tokenizer.decode(combined_ids, skip_special_tokens=True),
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            )["input_ids"]
            output_ids = self.t5_model.generate(input_ids)
            output_text = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_chunks.append(output_text)
        return "\n".join(output_chunks)

    def validate_report(self, original_text, generated_text):
        inputs = self.discriminator_tokenizer(original_text, generated_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.discriminator(**inputs)
        return torch.sigmoid(outputs.logits).tolist()

    def load_model(self, model_name):
        # Instead of file‑based loading, we query the DB for the model record.
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            return ModelLoadStatus.NOT_FOUND
        if record.status == ModelStatus.UNTRAINED.value:
            return ModelLoadStatus.UNTRAINED
        return ModelLoadStatus.SUCCESS

    def train(self, pdf_dfs_training, pdf_dfs_label, model_name, update_callback=None, bypass_already_trained=False, standard_format = ""):
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
                entity_data.append(self.extract_entities(text))

        label_data = []
        for df_index, df in enumerate(pdf_dfs_label):
            for text in df["Text"]:
                label_data.append(text)
                extracted_entities = self.extract_entities(text)
                for entity in extracted_entities:
                    if entity not in entity_data:
                        entity_data.append(entity)

        fine_tune_result = self.fine_tune(
            model_name=model_name,
            training_data=training_data,
            label_data=label_data,
            standard_format=standard_format,
            entity_data=entity_data,
            update_callback=update_callback
        )

        # Update the model record in the DB to mark it as TRAINED.
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            record = create_model_record(model_name)
        record.status = ModelStatus.TRAINED.value

        # Save the training configuration and history as JSON strings.
        record.train_config = json.dumps(fine_tune_result["hyperparameters"])
        record.train_history = json.dumps(fine_tune_result["loss_history"])

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
        # For each entity (if fine_tune_result returns a list of dicts with entity_type and entity_value),
        # add a new record to the NamedEntity table.
        for entity in fine_tune_result.get("named_entities", []):
            new_entity = NamedEntity(
                model_system_id=record.id,
                entity_type=entity.get("entity_type"),
                entity_value=entity.get("entity_value")
            )
            db.session.add(new_entity)

        # Update the TrainingData table if you have metadata (e.g., file names, file paths, processed_at)
        # For example, if you have a list of dictionaries with this info, iterate and add them.
        for file in pdf_dfs:
            file_name = file["filename"] if "filename" in file.columns else "unknown"
            file_path = f"uploads/{file_name}"
            processed_at = datetime.utcnow()

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
        return ModelStatus.TRAINED.value

    def process_pdfs(self, pdf_dfs, model_name, update_callback=None):
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
                update_callback(f"Processing PDF {df_index+1} of {len(pdf_dfs)}...")
            for text_index, text in enumerate(df["Text"]):
                if update_callback:
                    update_callback(f"Processing page {text_index+1}...")
                standardized_text = self.generate_standardized_report(text)
                validated_score = self.validate_report(text, standardized_text)
                standardized_report.append({
                    "filename": df["filename"].iloc[0],
                    "page_number": df["Page"].iloc[text_index],
                    "original_text": text,
                    "standardized_text": standardized_text,
                    "confidence": validated_score
                })
            reports.append(pd.DataFrame(standardized_report))
        if update_callback:
            update_callback("Processing completed. Standardized data saved.")
        return ModelLoadStatus.SUCCESS, reports
