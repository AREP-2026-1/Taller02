import json
import numpy as np
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 1. Cargar modelo
def model_fn(model_dir):
    with open(os.path.join(model_dir, "heart_disease_model.json"), "r") as f:
        model_dict = json.load(f)
    return model_dict

# 2. Procesar input
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError("Unsupported content type")

# 3. Predicción
def predict_fn(patient_features_raw, model_dict):
    bias = model_dict["bias"]
    weights = np.array(model_dict["weights"])
    norm_min = np.array(model_dict["normalization"]["min"])
    norm_range = np.array(model_dict["normalization"]["range"])

    features = [patient_features_raw[feat] for feat in model_dict["feature_names"]]
    features = np.array(features).reshape(1, -1)

    features_norm = (features - norm_min) / norm_range
    features_with_bias = np.c_[np.ones(1), features_norm]

    prob = sigmoid(features_with_bias @ np.concatenate([[bias], weights]))[0]
    prediction = 1 if prob >= 0.5 else 0

    return {
        "probability": float(prob),
        "prediction": prediction,
        "risk_level": "High" if prediction == 1 else "Low"
    }

# 4. Output
def output_fn(prediction, content_type):
    return json.dumps(prediction)
