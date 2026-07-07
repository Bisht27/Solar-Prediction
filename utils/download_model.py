import os
import gdown

MODEL_DIR = "model"
MODEL_NAME = "best_random_forest.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

FILE_ID = "1q72eX1nBmb7TljPgMAr91cZsPe6yuHir"


def download_model():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Downloading trained model...")

    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False,
    )

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed.")

    return MODEL_PATH