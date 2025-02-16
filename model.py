import os
import pickle
import re
import time

# This is static because it just creates a dummy pickle shell and does not modify currently loaded model
# Model must be trained to be loaded, this essentially just creates a temp placeholder file that tells the model where
# to save to
def create_model_pkl(model_name=""):
    if model_name == "":
        pattern = re.compile(r"model-(\d+)\.pkl")
        model_files = [f for f in os.listdir("models") if pattern.match(f)]
        next_number = max([int(pattern.match(f).group(1)) for f in model_files], default=-1) + 1
        model_name = f"model-{next_number}.pkl"
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        dummy_data = {"message": "This is a dummy pickle file."}
        with open(model_path, "wb") as f:
            pickle.dump(dummy_data, f)
    else:
        model_path = "File already exists"
    return model_path

class PDFStandardizer:
    current_loaded_model = None
    current_loaded_model_name = ""

    # Right now this just reads the pickle and reports error codes
    # will actually return model information plus error code when finished
    def load_model(self, model_name="model-1.pkl"):
        self.current_loaded_model_name = model_name
        model_path = os.path.join("models", model_name)

        if not os.path.exists(model_path):
            return 2  # File does not exist
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            print(data.get("message"))
            if isinstance(data, dict) and data.get("message") == "This is a dummy pickle file.":
                return 3  # File is a dummy pickle file
            return 1  # File exists and is a valid pickle file (not dummy)
        except (pickle.UnpicklingError, EOFError, AttributeError, TypeError):
            return 4  # File exists but is not a valid pickle file

    def train(self, pdf_dfs, model_name="", update_callback=None):
        model_path = os.path.join("models", model_name)
        if self.current_loaded_model_name == "":
            self.current_loaded_model = self.load_model(model_name)
        elif self.current_loaded_model_name != model_name:
            self.current_loaded_model = self.load_model(model_name)

        if update_callback:
            update_callback(f"Training started for {model_name}...")

        # Simulate model training
        for i in range(1, 6):  # Simulating training updates
            time.sleep(1)  # Simulate processing time
            if update_callback:
                update_callback(f"Step {i}/5: Processing data...")

        if not os.path.exists(model_path):
            dummy_data = {"message": "This is a trained pickle file."}
            with open(model_path, "wb") as f:
                pickle.dump(dummy_data, f)
        if update_callback:
            update_callback(f"Training completed. Model saved as {model_name}")

        return

    def process_pdfs(self, pdf_dfs, model_name="model-1.pkl", update_callback=None):
        if self.current_loaded_model_name == "":
            self.current_loaded_model = self.load_model(model_name)
        elif self.current_loaded_model_name != model_name:
            self.current_loaded_model = self.load_model(model_name)

        if self.current_loaded_model == 1:
            if update_callback:
                update_callback(f"Processing started for files using {model_name}...")

            # Simulate model training
            for i in range(1, 6):  # Simulating training updates
                time.sleep(1)  # Simulate processing time
                if update_callback:
                    update_callback(f"Step {i}/5: Processing pdf...")

            if update_callback:
                update_callback(f"Processing completed. Standardized data saved.")
        elif self.current_loaded_model == 3:
            if update_callback:
                update_callback(f"{model_name} has not been trained.")
        elif self.current_loaded_model == 4:
            if update_callback:
                update_callback(f"An unknown error has occurred with {model_name}")

        return self.current_loaded_model


model = PDFStandardizer()
model.train(None, "model-3.pkl")
