"""
Flask API of the SMS Spam detection model model.
"""
import joblib
import tarfile
from pathlib import Path
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import os
import urllib.request # For downloading models

from text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_URL = os.getenv("MODEL_URL")  # URL to model-release.tar.gz
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")

MODEL_PATH = MODEL_DIR / MODEL_FILE

def download(url: str, dest: Path):
    print(f"Downloading from {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def extract_tgz(tgz_path: Path, dest_dir: Path):
    print(f"Extracting {tgz_path} -> {dest_dir}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

# Priority 1: MODEL_URL (release asset) -> download once (cache) -> extract once -> find joblib
if MODEL_URL:
    tgz_path = MODEL_DIR / "model-release.tar.gz"
    extracted_marker = MODEL_DIR / ".extracted_ok"

    # Download only if not already cached
    if not tgz_path.is_file():
        download(MODEL_URL, tgz_path)
    else:
        print(f"Using cached artifact at {tgz_path}; skipping download.")

    # Extract only once (because /models is a volume)
    if not extracted_marker.is_file():
        extract_tgz(tgz_path, MODEL_DIR)
        extracted_marker.write_text("ok\n", encoding="utf-8")
    else:
        print(f"Extraction already done ({extracted_marker}); skipping extract.")

    # Common locations after extraction:
    candidate_paths = [
        MODEL_DIR / MODEL_FILE,
        MODEL_DIR / "outputs" / MODEL_FILE,
        MODEL_DIR / "output" / MODEL_FILE,
    ]
    found = next((p for p in candidate_paths if p.is_file()), None)
    if not found:
        raise FileNotFoundError(
            f"MODEL_URL was set but {MODEL_FILE} not found after extraction. "
            f"Tried: {candidate_paths}"
        )
    MODEL_PATH = found
    print(f"Using model from extracted artifact: {MODEL_PATH}")

# Priority 2: local/volume file at /models/model.joblib
elif MODEL_PATH.is_file():
    print(f"Using existing model at {MODEL_PATH}")

# Priority 3 (optional fallback): hardcoded default model (downloads into /models cache)
else:
    print(f"[WARNING]: {MODEL_FILE} not found and MODEL_URL not set.")
    tag = "model-20251119230101"
    asset_name = MODEL_FILE  # keep consistent with MODEL_FILE env var
    pre_existing_model_url = (
        f"https://github.com/doda25-team16/model-service/releases/download/{tag}/{asset_name}"
    )
    download(pre_existing_model_url, MODEL_PATH)
    print(f"Downloaded default model into {MODEL_PATH}")

model = joblib.load(str(MODEL_PATH))
print(f"Loaded model from {MODEL_PATH}")

print(f"Loaded model from {MODEL_PATH}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json(silent=True) or {}
    sms = input_data.get('sms')

    if sms is None:
        return jsonify({"error": "Missing required field 'sms'"}), 400
    if not isinstance(sms, str):
        return jsonify({"error": "'sms' must be a string"}), 400

    processed_sms = prepare(sms)
    prediction = model.predict(processed_sms)[0]

    res = {"result": prediction, "classifier": "decision tree", "sms": sms}
    return jsonify(res)


if __name__ == '__main__':
    #clf = joblib.load('output/model.joblib')
    port = int(os.getenv("MODEL_PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=False)
