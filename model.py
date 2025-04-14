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

        # -------------------------------------------------
        # Generator: flan-t5-large
        # -------------------------------------------------
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        # -------------------------------------------------
        # Discriminator: BART MNLI (3-class NLI)
        # - contradiction=0, neutral=1, entailment=2
        # -------------------------------------------------
        self.discriminator = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        self.discriminator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

        # Place models on device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.discriminator.to(self.device)

        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        # Example patterns
        patterns = [
            {"label": "SOIL_TYPE", "pattern": [{"LOWER": "soil"}, {"LOWER": "type"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "sandy", "OP": "?"}, {"LOWER": "loam", "OP": "?"}]},
            {"label": "ORDER_NUMBER", "pattern": [{"LOWER": "order"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "#", "OP": "?"}, {"IS_ALPHA": False, "OP": "+"}]},
            {"label": "TEST_RESULT", "pattern": "Contamination Test: Passed"}
            # etc.
        ]
        ruler.add_patterns(patterns)

    # -------------------------------------------------------------------------
    # 1. Helper for verifying presence of "must-be-correct" entities in text
    # -------------------------------------------------------------------------
    def verify_critical_entities(self, generated_text: str, critical_entities: list) -> bool:
        """
        Returns True if *all* critical_entities appear verbatim
        in the generated_text. Otherwise returns False.
        """
        for entity in critical_entities:
            if entity not in generated_text:
                return False
        return True

    # -------------------------------------------------------------------------
    # 2. Logic for labeling NLI classes (entailment=2, contradiction=0)
    #    We skip neutral=1 for simplicity
    # -------------------------------------------------------------------------
    def label_nli_class(self, gold_text: str, generated_text: str, critical_entities: list) -> int:
        """
        Return NLI label for a single example:
          2 => entailment
          0 => contradiction
        Currently skipping 'neutral' (1).

        - If ANY critical entity is missing => contradiction
        - If all entities appear AND the generated text exactly matches gold_text => entailment
        - Otherwise => contradiction
        """
        # Step 1: Must contain all critical named entities
        has_all_ents = self.verify_critical_entities(generated_text, critical_entities)
        if not has_all_ents:
            return 0  # contradiction

        # Step 2: Optionally check exact match
        if generated_text.strip() == gold_text.strip():
            return 2  # entailment
        else:
            return 0  # contradiction

    # -------------------------------------------------------------------------
    # 3. Extract a list of "critical" entity strings from your entity_data
    #    For demonstration: we treat *all* discovered entities as critical
    # -------------------------------------------------------------------------
    def get_critical_entities(self, entity_list):
        """
        entity_list is something like:
            [{"entity": "Acme Corp", "type": "ORG"}, {"entity": "1937", "type": "DATE"}, ...]
        Return them as a list of strings that must appear verbatim in the text.
        """
        return [e["entity"] for e in entity_list]

    def fine_tune(self, model_name, training_data, label_data, standard_format, entity_data=None,
                  epochs=1, batch_size=1, lr=5e-5, update_callback=None):
        """
        Fine tune the T5 model using training_data and label_data.
        Optionally, entity_data can be concatenated to the input for context.
        The blank_format is used as a prompt template.

        Now using NLI-based classification for the discriminator:
        - If all critical entities appear AND generated_text == gold => 'entailment' (2)
        - Otherwise => 'contradiction' (0)
        """
        optimizer = Adam(self.model.parameters(), lr=lr)

        # Generator loss (unchanged T5 cross-entropy)
        ce_loss_generator = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Discriminator NLI loss (3-way cross entropy: contradiction=0, neutral=1, entailment=2)
        nli_loss = CrossEntropyLoss()

        dataset_size = len(training_data)
        self.model.train()
        self.discriminator.train()

        loss_history = []
        last_gen_loss = None
        last_disc_loss = None

        for epoch in range(epochs):
            epoch_losses = []
            if update_callback:
                update_callback(f"Epoch {epoch + 1}/{epochs} started.")
            for i in range(0, dataset_size, batch_size):
                batch_inputs = []
                batch_labels = []
                batch_start_index = i

                for j in range(i, min(i + batch_size, dataset_size)):
                    entity_info = ""
                    if entity_data:
                        entity_info = " | Entities: " + str(entity_data[j])

                    prompt = (
                        f"Output Format: {standard_format} "
                        f"and Named Entities: {entity_info} "
                        f"with Input: {training_data[j]}. "
                        "If any factual information is missing, output '{CANNOT FIND}' in its place."
                    )
                    batch_inputs.append(prompt)
                    batch_labels.append(label_data[j])

                # Tokenize inputs and labels
                input_encodings = self.tokenizer(batch_inputs, return_tensors="pt",
                                                 padding=True, truncation=True, max_length=512)
                label_encodings = self.tokenizer(batch_labels, return_tensors="pt",
                                                 padding=True, truncation=True, max_length=512)
                input_ids = input_encodings.input_ids.to(self.device)
                attention_mask = input_encodings.attention_mask.to(self.device)
                labels = label_encodings.input_ids.to(self.device)

                # ------------------------------
                # Generator forward pass
                # ------------------------------
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                gen_loss = outputs.loss  # T5 cross-entropy (already computed internally)

                # ------------------------------
                # Generate text for discriminator
                # ------------------------------
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512
                    )
                generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

                # Prepare NLI labels for the discriminator
                disc_labels_list = []
                for j in range(i, min(i + batch_size, dataset_size)):
                    gold_text = label_data[j]
                    gen_text = generated_texts[j - batch_start_index]
                    crit_ents = []
                    # If entity_data is available, gather the "critical" named entities for this sample
                    if entity_data:
                        crit_ents = self.get_critical_entities(entity_data[j])

                    nli_label = self.label_nli_class(
                        gold_text=gold_text,
                        generated_text=gen_text,
                        critical_entities=crit_ents
                    )
                    disc_labels_list.append(nli_label)

                disc_labels = torch.tensor(disc_labels_list, dtype=torch.long, device=self.device)

                # Now feed premise (original input) & hypothesis (generated text)
                premise_texts = training_data[i : i + len(disc_labels_list)]
                hypothesis_texts = generated_texts

                disc_inputs = self.discriminator_tokenizer(
                    premise_texts,
                    hypothesis_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                disc_inputs = {k: v.to(self.device) for k, v in disc_inputs.items()}
                disc_outputs = self.discriminator(**disc_inputs)

                # NLI classification loss
                disc_loss = nli_loss(disc_outputs.logits, disc_labels)

                # Total combined loss
                total_loss = gen_loss + 0.5 * disc_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())
                last_gen_loss = gen_loss.item()
                last_disc_loss = disc_loss.item()

                if update_callback:
                    update_callback(
                        f"Batch {i // batch_size + 1}: gen_loss={last_gen_loss:.4f}, disc_loss={last_disc_loss:.4f}"
                    )

            loss_history.append({
                "epoch": epoch + 1,
                "average_loss": sum(epoch_losses) / len(epoch_losses),
                "batch_losses": epoch_losses
            })

        generator_weights_path = f"models/{model_name}_finetuned.pt"
        discriminator_weights_path = f"models/{model_name}_finetuned.pt"
        self.model.save_pretrained(generator_weights_path)
        self.discriminator.save_pretrained(discriminator_weights_path)

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

    def extract_entities(self, text, update_callback):
        doc = self.nlp(text)
        all_entities = []

        for ent in doc.ents:
            all_entities.append({"entity": ent.text, "type": ent.label_})
        if update_callback:
            update_callback("Extracting entities...")
            update_callback(f"Entities found: {len(all_entities)}")
            update_callback(f"Entities: {all_entities}")
        return all_entities

    def generate_standardized_report(self, text, max_input_length=512, chunk_overlap=0):
        """
        We add a simple 'reject-or-retry' if the discriminator says 'contradiction' (label=0).
        For demonstration, we try up to 3 times.
        """
        prefix = "standardize: "
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
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
            # We'll do up to 3 attempts if the discriminator sees contradiction
            attempts = 0
            best_text = None
            while attempts < 3:
                input_ids = self.tokenizer(
                    self.tokenizer.decode(combined_ids, skip_special_tokens=True),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_length
                )["input_ids"].to(self.device)

                output_ids = self.model.generate(input_ids)
                candidate_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Check if the discriminator sees contradiction
                nli_pred = self._predict_nli_class(
                    premise=self.tokenizer.decode(chunk, skip_special_tokens=True),
                    hypothesis=candidate_text
                )
                if nli_pred == 0:
                    # label=0 => contradiction
                    attempts += 1
                else:
                    # label=2 => entailment or label=1 => neutral => accept
                    best_text = candidate_text
                    break

            if best_text is None:
                # if still contradiction after attempts, accept last candidate to avoid infinite loop
                best_text = candidate_text

            output_chunks.append(best_text)

        return "\n".join(output_chunks)

    def _predict_nli_class(self, premise, hypothesis):
        """
        Helper to get the predicted NLI label from the discriminator:
        0 => contradiction
        1 => neutral
        2 => entailment
        """
        inputs = self.discriminator_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        logits = self.discriminator(**inputs).logits
        return torch.argmax(logits, dim=1).item()

    def validate_report(self, original_text, generated_text):
        """
        For backward compatibility, we keep this method returning a list of raw
        class probabilities. But now it's a 3-way distribution for (contradiction, neutral, entailment).
        """
        inputs = self.discriminator_tokenizer(
            original_text,
            generated_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        outputs = self.discriminator(**inputs)
        # Return softmax distribution over [contradiction, neutral, entailment]
        probs = torch.softmax(outputs.logits, dim=1)
        return probs.detach().cpu().tolist()

    def load_model(self, model_name):
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            return ModelLoadStatus.NOT_FOUND
        if record.status == ModelStatus.UNTRAINED.value:
            return ModelLoadStatus.UNTRAINED
        return ModelLoadStatus.SUCCESS

    def train(self, pdf_dfs_training, pdf_dfs_label, model_name, update_callback=None, bypass_already_trained=False, standard_format=""):
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
                extracted_entities = self.extract_entities(text, update_callback)
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
        discriminator_llm.architecture = "bart-large-mnli"
        discriminator_llm.weights_path = fine_tune_result["discriminator_weights_path"]
        discriminator_llm.hyperparameters = json.dumps(fine_tune_result["hyperparameters"])
        discriminator_llm.train_status = TrainStatus.TRAINED.value
        db.session.add(discriminator_llm)

        # Update Named Entities
        for entity_group in fine_tune_result.get("named_entities", []):
            if isinstance(entity_group, list):
                for entity in entity_group:
                    if isinstance(entity, dict):
                        new_entity = NamedEntity(
                            model_system_id=record.id,
                            entity_type=entity.get("type"),
                            entity_value=entity.get("entity")
                        )
                        db.session.add(new_entity)
            elif isinstance(entity_group, dict):
                new_entity = NamedEntity(
                    model_system_id=record.id,
                    entity_type=entity_group.get("type"),
                    entity_value=entity_group.get("entity")
                )
                db.session.add(new_entity)

        # Update the TrainingData table if you have metadata
        for file in pdf_dfs_training:
            file_name = file["filename"].iloc[0] if "filename" in file.columns else "unknown"
            file_path = f"uploads/{file_name}"
            processed_at = datetime.utcnow()

            new_td = TrainingData(
                model_system_id=record.id,
                file_name=file_name,
                file_path=file_path,
                processed_at=processed_at
            )
            db.session.add(new_td)

        for file in pdf_dfs_label:
            file_name = file["filename"].iloc[0] if "filename" in file.columns else "unknown"
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