import os
import pickle
import re
import time
import spacy
from enum import Enum
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
import pandas as pd
from werkzeug.utils import secure_filename


class ModelLoadStatus(Enum):
    SUCCESS = 1
    FILE_NOT_FOUND = 2
    UNTRAINED = 3
    CORRUPTED = 4

# This is static because it just creates a dummy pickle shell and does not modify currently loaded model
# Model must be trained to be loaded, this essentially just creates a temp placeholder file that tells the model where
# to save to
def create_model_pkl(model_name=""):
    if model_name == "":
        pattern = re.compile(r"model-(\d+)\.pkl")
        model_files = [f for f in os.listdir("models") if pattern.match(f)]
        next_number = max([int(pattern.match(f).group(1)) for f in model_files], default=-1) + 1
        model_name = f"model-{next_number}.pkl"
    model_name = secure_filename(model_name)
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        dummy_data = {"message": "This is a dummy pickle file."}
        with open(model_path, "wb") as f:
            pickle.dump(dummy_data, f)
    return model_path

class PDFStandardizer:

    current_loaded_model = None
    current_loaded_model_name = ""

    def __init__(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        self.nlp = spacy.load("en_core_web_sm")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.discriminator = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.discriminator_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
            chunk = text_ids[start:end]
            chunks.append(chunk)
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
        inputs = self.discriminator_tokenizer(original_text, generated_text, return_tensors="pt", padding=True,
                                              truncation=True)
        outputs = self.discriminator(**inputs)
        return torch.sigmoid(outputs.logits).tolist()

    def load_model(self, model_name="model-1.pkl"):
        model_path = os.path.join("models", model_name)

        if not os.path.exists(model_path):
            return ModelLoadStatus.FILE_NOT_FOUND  # File does not exist
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            self.current_loaded_model_name = model_name
            self.current_loaded_model = model_data
            if isinstance(model_data, dict) and model_data.get("message") == "This is a dummy pickle file.":
                return ModelLoadStatus.UNTRAINED  # Model exists but is not trained
            return ModelLoadStatus.SUCCESS  # Valid trained model
        except (pickle.UnpicklingError, EOFError, AttributeError, TypeError):
            return ModelLoadStatus.CORRUPTED  # Corrupted or invalid file

    def train(self, pdf_dfs, model_name="model-1.pkl", update_callback=None, bypass_already_trained=False):
        if (self.current_loaded_model_name == model_name
                and not (isinstance(self.current_loaded_model, dict)
                 and self.current_loaded_model.get("message") == "This is a dummy pickle file.")):
            error = ModelLoadStatus.SUCCESS
        else:
            error = self.load_model(model_name)

        if error == ModelLoadStatus.UNTRAINED or (bypass_already_trained and error == ModelLoadStatus.SUCCESS):
            if update_callback:
                update_callback(f"Training started for {model_name}...")

            entity_data = []
            for df_index, df in enumerate(pdf_dfs):
                if update_callback:
                    update_callback(f"Training on pdf {df_index+1} of {len(pdf_dfs)}...")
                for text_index, text in enumerate(df["Text"]):
                    entity_data.append(self.extract_entities(text))
                    if update_callback:
                        s = len(df["Text"])
                        update_callback(f"Training on page {text_index+1} of {s}...")

            trained_model = {"entities": entity_data}
            model_path = os.path.join("models", model_name)
            with open(model_path, "wb") as f:
                pickle.dump(trained_model, f)

            error = self.load_model(model_name)

            if update_callback and error == ModelLoadStatus.SUCCESS:
                update_callback(f"Training completed. Model saved as {model_name}")
            else:
                update_callback(f"Error on saving {model_name} after training")

        return error

    def process_pdfs(self, pdf_dfs, model_name="model-1.pkl", update_callback=None):
        reports = []
        if self.current_loaded_model_name == model_name and not (isinstance(self.current_loaded_model, dict)
                 and self.current_loaded_model.get("message") == "This is a dummy pickle file."):
            error = ModelLoadStatus.SUCCESS
        else:
            error = self.load_model(model_name)

        if error == ModelLoadStatus.SUCCESS:
            if update_callback:
                update_callback(f"Processing started for files using {model_name}...")

            for df_index, df in enumerate(pdf_dfs):
                standardized_report = []
                if update_callback:
                    update_callback(f"Processing pdf {df_index+1} of {len(pdf_dfs)}...")
                s = len(df["Text"])
                for text_index, text in enumerate(df["Text"]):
                    if update_callback:
                        update_callback(f"Processing page {text_index+1} of {s}...")
                    standardized_text = self.generate_standardized_report(text)
                    validated_score = self.validate_report(text, standardized_text)
                    standardized_report.append({
                        "filename": df["filename"].iloc[0],
                        "page_number": df["Page"].iloc[text_index],
                        "original_text": text,
                        "standardized_text": standardized_text,
                        "confidence": validated_score
                    })
                standardized_report = pd.DataFrame(standardized_report)
                reports.append(standardized_report)

            if update_callback and reports:
                update_callback(f"Processing completed. Standardized data saved.")
            else:
                update_callback(f"Something went wrong while processing files")

        return error, reports
