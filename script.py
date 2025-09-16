import os

folders = [
    "data/raw", "data/interim", "data/processed",
    "models", "notebooks", "configs",
    "src/data", "src/features", "src/models", "src/serve", "src/utils",
    "tests"
]

files = [
    "README.md", ".gitignore", "requirements.txt", "Dockerfile",
    "configs/config.yaml", "configs/data.yaml", "configs/model.yaml", "configs/train.yaml"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, "a").close()

print("Project structure created ✅")