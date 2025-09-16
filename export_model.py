import os
import shutil

export_dir = "models"
MODEL_SOURCE_FILE = "mlruns/0/models/m-8b0af3c7119743bb8631f86f0904f04f/artifacts/data/model.pth"

os.makedirs(export_dir, exist_ok=True)
dest_model_file = os.path.join(export_dir, "model.pth")

if os.path.exists(MODEL_SOURCE_FILE):
    shutil.copy(MODEL_SOURCE_FILE, dest_model_file)
    print(f"Copied model to: {dest_model_file}")
else:
    print(f"Error: Model file not found at {MODEL_SOURCE_FILE}")