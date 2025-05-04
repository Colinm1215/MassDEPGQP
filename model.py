import json
import os
import random
from datetime import datetime

from torch import softmax
from torch.utils.data import TensorDataset, DataLoader

from schema_utils import load_schema, json_to_xlsx
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
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from pdfScript import get_clean_dataframe
import random, sys, time, torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
import jsonschema
from torch.nn.functional import softmax



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
        if not os.path.exists("formats"):
            os.makedirs("formats")

        self.nlp = spacy.load("en_core_web_trf")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.discriminator = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        self.discriminator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_model.to(self.device)
        self.discriminator.to(self.device)
        self.standard_format = None
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = [
            {"label": "SOIL_TYPE",
             "pattern": [
                 {"LOWER": "soil"}, {"LOWER": "type"}, {"IS_PUNCT": True, "OP": "?"},
                 {"LOWER": {"IN": ["sandy", "clay", "loam", "silt", "gravel"]}, "OP": "+"}
             ]},
            {"label": "ORDER_NUMBER",
             "pattern": [
                 {"LOWER": "order"}, {"IS_PUNCT": True, "OP": "?"},
                 {"LOWER": "#", "OP": "?"}, {"TEXT": {"REGEX": r"[A-Za-z0-9\-]+"}, "OP": "+"}
             ]},
            {"label": "TEST_RESULT",
             "pattern": [
                 {"LOWER": "test"}, {"TEXT": ":", "OP": "?"},
                 {"TEXT": {"REGEX": r".+"}},
             ]},

            {"label": "DATE",
             "pattern": [{"TEXT": {"REGEX": r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"}}]},

            {"label": "SOURCE_LOCATION",
             "pattern": [
                 {"LOWER": "source"}, {"LOWER": "location"}, {"TEXT": ":", "OP": "?"},
                 {"TEXT": {"REGEX": r".+"}},
             ]},

            {"label": "DESTINATION",
             "pattern": [
                 {"LOWER": "destination"}, {"TEXT": ":", "OP": "?"},
                 {"TEXT": {"REGEX": r".+"}},
             ]},

            {"label": "ADDRESS",
             "pattern": [
                 {"LIKE_NUM": True}, {"IS_ALPHA": True, "OP": "+"},
                 {"LOWER": {"IN": ["ave", "street", "road", "blvd", "drive", "lane"]}, "OP": "?"}
             ]},

            {"label": "VOLUME",
             "pattern": [
                 {"LIKE_NUM": True}, {"LOWER": {"IN": ["cubic", "cu"]}}, {"LOWER": {"IN": ["yard", "yards"]}}
             ]},

            {"label": "SAFETY_STATUS",
             "pattern": [
                 {"LOWER": "safety"}, {"LOWER": "status"}, {"TEXT": ":", "OP": "?"},
                 {"TEXT": {"REGEX": r".+"}}
             ]},

            {"label": "SPECIAL_HANDLING",
             "pattern": [
                 {"LOWER": "special"}, {"LOWER": "handling"}, {"TEXT": ":", "OP": "?"},
                 {"TEXT": {"REGEX": r".+"}}
             ]},

            {"label": "PHONE_NUMBER",
             "pattern": [{"TEXT": {"REGEX": r"\(\d{3}\)\s*\d{3}-\d{4}"}}]},
        ]
        ruler.add_patterns(patterns)

    def _looks_like_json(self, txt: str) -> bool:
        try:
            json.loads(txt)
            return True
        except Exception:
            return False


    def fine_tune(
        self,
        model_name,
        training_data,
        label_data,
        standard_format,
        entity_data=None,
        epochs: int = 3,
        batch_size: int = 1,
        lr: float = 1e-5,
        update_callback=None,
    ):
        """
        Trains generator and discriminator:
          1. Supervised cross-entropy for format learning.
          2. Adversarial training using BART-MNLI entailment.

        Signature and return interface remain unchanged.
        """

        # Load or use provided JSON schema
        schema = (
            standard_format
            if isinstance(standard_format, dict)
            else load_schema(standard_format)
        )

        pages = training_data
        gold = label_data

        # Build record->page mapping
        page_indices = []
        for idx, rec_txt in enumerate(gold):
            if self._looks_like_json(rec_txt):
                rec = json.loads(rec_txt)
                key_vals = [str(v) for v in rec.values()]
            else:
                key_vals = [rec_txt.split("\n", 1)[0].strip()]

            matched = False
            for pi, text in enumerate(pages):
                if any(kv in text for kv in key_vals if kv):
                    page_indices.append(pi)
                    matched = True
                    break

            if not matched:
                if update_callback:
                    update_callback(
                        f"[WARN] No page match for record {idx}, defaulting to page 0"
                    )
                page_indices.append(0)

        # Tokenize pages and labels
        enc_pages = self.t5_tokenizer(
            pages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc_gold = self.t5_tokenizer(
            gold,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        # Supervised DataLoader
        sup_idx = torch.tensor(page_indices, dtype=torch.long)
        sup_inputs = enc_pages.input_ids.index_select(0, sup_idx).to(self.device)
        sup_masks = enc_pages.attention_mask.index_select(0, sup_idx).to(
            self.device
        )
        sup_targets = enc_gold.input_ids.to(self.device)
        sup_loader = DataLoader(
            TensorDataset(sup_inputs, sup_masks, sup_targets),
            batch_size=batch_size,
            shuffle=True,
        )

        # Adversarial DataLoader
        unsup_loader = DataLoader(
            TensorDataset(torch.arange(len(gold))),
            batch_size=batch_size,
            shuffle=True,
        )

        if update_callback:
            update_callback(
                f"[DEBUG] sup_batches={len(sup_loader)}, unsup_batches={len(unsup_loader)}"
            )

        # Optimizers and loss
        gen_opt = AdamW(self.t5_model.parameters(), lr=lr)
        disc_opt = AdamW(self.discriminator.parameters(), lr=lr * 0.5)
        adv_loss_fn = BCEWithLogitsLoss()

        lambda_adv = 0.1
        gen_steps_per_disc = 2
        gen_kwargs = {"max_length": 64, "num_beams": 1}

        loss_history = []
        last_gen_loss = last_disc_loss = 0.0

        # Training loop
        for epoch in range(1, epochs + 1):
            # Supervised phase
            sup_acc = 0.0
            for in_ids, in_mask, tgt_ids in sup_loader:
                in_ids, in_mask, tgt_ids = (
                    in_ids.to(self.device),
                    in_mask.to(self.device),
                    tgt_ids.to(self.device),
                )
                gen_opt.zero_grad()
                loss_sup = self.t5_model(
                    input_ids=in_ids,
                    attention_mask=in_mask,
                    labels=tgt_ids,
                ).loss
                loss_sup.backward()
                clip_grad_norm_(self.t5_model.parameters(), 1.0)
                gen_opt.step()
                sup_acc += loss_sup.item()

            avg_sup = sup_acc / len(sup_loader)
            if update_callback:
                update_callback(
                    f"[DEBUG] Epoch {epoch} SUP avg_loss={avg_sup:.6f}"
                )

            # Adversarial phase
            adv_acc = disc_acc = 0.0
            for (idx_tensor,) in unsup_loader:
                idx = idx_tensor.item()
                pi = page_indices[idx]
                page_text = pages[pi]

                # Generate once for discriminator
                gen_ids = self.t5_model.generate(
                    input_ids=enc_pages.input_ids[pi : pi + 1].to(self.device),
                    attention_mask=
                        enc_pages.attention_mask[pi : pi + 1].to(self.device),
                    **gen_kwargs,
                )
                fake_txt = self.t5_tokenizer.decode(
                    gen_ids[0], skip_special_tokens=True
                )

                # Schema enforcement
                try:
                    parsed = json.loads(fake_txt)
                    jsonschema.validate(parsed, schema)
                except Exception:
                    pass

                real_txt = gold[idx]

                # Tokenize for discriminator (using entailment)
                real_in = self.discriminator_tokenizer(
                    page_text,
                    real_txt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)
                fake_in = self.discriminator_tokenizer(
                    page_text,
                    fake_txt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                # Discriminator update
                disc_opt.zero_grad()
                logits_r = self.discriminator(**real_in).logits[:, 2]
                logits_f = self.discriminator(**fake_in).logits[:, 2]
                loss_d = (
                    adv_loss_fn(logits_r, torch.ones_like(logits_r)) +
                    adv_loss_fn(logits_f, torch.zeros_like(logits_f))
                ) * 0.5
                loss_d.backward()
                clip_grad_norm_(self.discriminator.parameters(), 1.0)
                disc_opt.step()

                disc_acc += loss_d.item()
                last_disc_loss = loss_d.item()

                # Generator adversarial updates
                for _ in range(gen_steps_per_disc):
                    gen_opt.zero_grad()
                    # Regenerate after update
                    gen_ids2 = self.t5_model.generate(
                        input_ids=enc_pages.input_ids[pi : pi + 1].to(self.device),
                        attention_mask=
                            enc_pages.attention_mask[pi : pi + 1].to(self.device),
                        **gen_kwargs,
                    )
                    fake_txt2 = self.t5_tokenizer.decode(
                        gen_ids2[0], skip_special_tokens=True
                    )
                    fake_in2 = self.discriminator_tokenizer(
                        page_text,
                        fake_txt2,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    ).to(self.device)
                    logits_f2 = self.discriminator(**fake_in2).logits[:, 2]
                    loss_g = adv_loss_fn(logits_f2, torch.ones_like(logits_f2))
                    scaled_loss_g = lambda_adv * loss_g
                    scaled_loss_g.backward()
                    clip_grad_norm_(self.t5_model.parameters(), 1.0)
                    gen_opt.step()

                    adv_acc += scaled_loss_g.item()
                    last_gen_loss = scaled_loss_g.item()

            avg_adv = adv_acc / len(unsup_loader)
            avg_disc = disc_acc / len(unsup_loader)
            loss_history.append({
                "epoch": epoch,
                "sup": avg_sup,
                "adv": avg_adv,
                "disc": avg_disc,
            })

            if update_callback:
                update_callback(
                    f"[DEBUG] Epoch {epoch} ADV avg={avg_adv:.6f}, DISC avg={avg_disc:.6f}"
                )

        # Save models
        gen_path = f"models/{model_name}_finetuned_generator"
        disc_path = f"models/{model_name}_finetuned_discriminator"
        self.t5_model.save_pretrained(gen_path)
        self.discriminator.save_pretrained(disc_path)

        return {
            "final_loss": last_gen_loss + last_disc_loss,
            "generator_loss": last_gen_loss,
            "discriminator_loss": last_disc_loss,
            "epochs": epochs,
            "loss_history": loss_history,
            "hyperparameters": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            },
            "generator_weights_path": gen_path,
            "discriminator_weights_path": disc_path,
            "training_data": training_data,
            "named_entities": entity_data,
            "standard_format": standard_format,
        }

    def extract_entities(self, text, update_callback):
        doc = self.nlp(text)
        all_entities = []

        for ent in doc.ents:
            all_entities.append({
                "entity_type": ent.label_,
                "entity_value": ent.text
            })

        if update_callback:
            update_callback(f"Extracted {len(all_entities)} entities: {all_entities}")

        return all_entities

    def generate_standardized_report(
            self,
            report_df: pd.DataFrame,
            model_name: str,
            update_callback=None
    ) -> str:
        """
        For a single PDF (as a DataFrame with columns 'filename', 'Page', 'Text', optionally 'Corrected_Text'),
        generates a standardized record per page, validates each with the discriminator,
        writes an .xlsx into processed/<model_name>/ and returns that file path.
        """
        filename = report_df["filename"].iloc[0]
        processed = []

        for _, row in report_df.iterrows():
            text = row.get("Corrected_Text", row["Text"])
            page_no = row["Page"]

            # --- 1) Generate JSON dict for this page ---
            prompt_entities = json.dumps(self.extract_entities(text, update_callback), ensure_ascii=False)
            prompt = (
                "You are a data-standardisation assistant.\n"
                "Return **only** a valid JSON object matching this schema:\n"
                f"{self.standard_format}\n\n"
                f"TEXT:\n{text}\n\n"
                f"Entities: {prompt_entities}"
            )
            enc = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen_ids = self.t5_model.generate(**enc, max_length=256, num_beams=2, early_stopping=True)
            gen_str = self.t5_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            try:
                std_dict = json.loads(gen_str)
            except json.JSONDecodeError:
                std_dict = {}
            if update_callback:
                update_callback(f"Page {page_no} generated JSON: {std_dict}")

            # --- 2) Validate factual consistency ---
            conf = self.validate_report(text, std_dict)
            if update_callback:
                update_callback(f"Page {page_no} confidence: {conf:.3f}")

            # --- 3) Assemble record ---
            record = {
                "filename": filename,
                "page_number": page_no,
                **std_dict,
                "confidence": conf,
            }
            processed.append(record)

        # --- 4) Dump to Excel ---
        out_dir = os.path.join("processed", model_name)
        os.makedirs(out_dir, exist_ok=True)
        xlsx_path = os.path.join(out_dir, f"{filename}_Standardized.xlsx")

        # Load your schema (either dict or filepath)
        schema = (
            self.standard_format
            if isinstance(self.standard_format, dict)
            else load_schema(self.standard_format)
        )
        json_to_xlsx(
            {next(iter(schema["sheets"])): processed},
            schema,
            xlsx_path
        )

        if update_callback:
            update_callback(f"Wrote Excel → {xlsx_path}")
        return xlsx_path

    def validate_report(self, original_text: str, generated: object) -> float:
        """
        Converts `generated` to JSON text if needed, then returns
        the entailment probability from BART-MNLI’s 'entailment' class.
        """
        gen_str = json.dumps(generated, ensure_ascii=False) if isinstance(generated, dict) else str(generated)
        inputs = self.discriminator_tokenizer(
            original_text,
            gen_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        logits = self.discriminator(**inputs).logits
        probs = softmax(logits, dim=-1)
        # label index 2 == 'entailment'
        return probs[0, 2].item()

    def load_model(self, model_name, update_callback=None):
        record = ModelSystem.query.filter_by(name=model_name).first()
        if not record:
            return ModelLoadStatus.NOT_FOUND
        if record.status == ModelStatus.UNTRAINED.value:
            return ModelLoadStatus.UNTRAINED

        try:
            path = record.standard_format or ""
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.standard_format = json.load(f)
            else:
                self.standard_format = {}
        except Exception:
            self.standard_format = {}

        generator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="generator").first()
        discriminator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="discriminator").first()

        try:
            if generator_llm and generator_llm.weights_path and os.path.exists(generator_llm.weights_path):
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(generator_llm.weights_path)
                self.t5_model.to(self.device)
            else:
                if update_callback:
                    update_callback(f"[ERROR] could not load weights for {model_name}")
                return ModelLoadStatus.ERROR

            if discriminator_llm and discriminator_llm.weights_path and os.path.exists(discriminator_llm.weights_path):
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
            if "Text" in df.columns:
                for text in df["Text"]:
                    label_data.append(text)
            else:
                records = df.fillna("").to_dict(orient="records")
                if update_callback:
                    update_callback(f"Label data is not in Text column, using records: {records}")
                for rec in records:
                    label_data.append(json.dumps(rec))

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
        schema = fine_tune_result["standard_format"]
        fmt_path = os.path.join("formats", f"{model_name}_schema.json")
        with open(fmt_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        record.standard_format = fmt_path

        # Update or create the generator LLM record.
        generator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="generator").first()
        if not generator_llm:
            generator_llm = LLMModel(model_system_id=record.id, llm_type="generator")
        generator_llm.architecture = "google/flan-t5-large"
        generator_llm.weights_path = fine_tune_result["generator_weights_path"]
        generator_llm.hyperparameters = json.dumps(fine_tune_result["hyperparameters"])
        generator_llm.train_status = TrainStatus.TRAINED.value
        db.session.add(generator_llm)

        # Update or create the discriminator LLM record.
        discriminator_llm = LLMModel.query.filter_by(model_system_id=record.id, llm_type="discriminator").first()
        if not discriminator_llm:
            discriminator_llm = LLMModel(model_system_id=record.id, llm_type="discriminator")
        discriminator_llm.architecture = "facebook/bart-large-mnli"
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

    def process_pdfs(
        self,
        pdf_dfs: list[pd.DataFrame],
        model_name: str,
        update_callback=None
    ):
        """
        Loops through each DataFrame in `pdf_dfs`, calls
        generate_standardized_report → Excel for each,
        and returns (status, [list of Excel file paths]).
        """
        status = self.load_model(model_name, update_callback)
        if status != ModelLoadStatus.SUCCESS:
            if update_callback:
                update_callback("Model must be trained first.")
            return status, None

        outputs = []
        for df in pdf_dfs:
            if update_callback:
                update_callback(f"Starting report for {df['filename'].iloc[0]}")
            excel_path = self.generate_standardized_report(df, model_name, update_callback)
            outputs.append(excel_path)

        return ModelLoadStatus.SUCCESS, outputs