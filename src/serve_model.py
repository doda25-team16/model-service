"""
Flask API of the SMS Spam detection model model.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
import os
import urllib.request # For downloading models

from text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)

# Logic for Assignment 1: F10 (default model or provided model)
#MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_DIR = "/models"
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Create directory & verify if file exists:
os.makedirs(MODEL_DIR, exist_ok=True)
file_exists = os.path.isfile(MODEL_PATH)

if not file_exists:
  print(f"[WARNING]: {MODEL_FILE} not found.")
  # No model provided via volume -> download default model from GitHub release (from our repo)
  # Example: https://github.com/doda25-team16/model-service/releases/download/model-20251119230101/model.joblib
  tag = "model-20251119230101"
  asset_name = 'model.joblib'
  pre_existing_model_url = f"https://github.com/doda25-team16/model-service/releases/download/{tag}/{asset_name}"

  urllib.request.urlretrieve(pre_existing_model_url, MODEL_PATH) # downloads model into file
  print(f"Model did NOT EXIST, downloading a default model from release into {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
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
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    prediction = model.predict(processed_sms)[0]
    
    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    #clf = joblib.load('output/model.joblib')
    port = int(os.getenv("MODEL_PORT", 8081))
    
    app.run(host="0.0.0.0", port=port, debug=True)
