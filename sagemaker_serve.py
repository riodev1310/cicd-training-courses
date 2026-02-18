import os
import json
import pickle
import numpy as np
import pandas as pd
import io
import flask
from flask import Flask, Response
from sklearn.preprocessing import LabelEncoder  # Thêm để encode categorical features nếu cần

app = flask.Flask(__name__)


# inference.py functions
def get_model(model_dir):
    '''Load model from artifact store (e.g., MLflow or local)'''
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def prepare_input(request_body, request_content_type):
    '''Process input data for HR model'''
    if request_content_type == 'application/json':
        data = pd.read_json(io.StringIO(request_body), orient='split')
        
        # Normalize column names to lowercase (như trong DAG training)
        data.columns = data.columns.str.lower()
        
        # Handle missing values (như trong training)
        data = data.fillna(0)
        
        # Encode categorical columns (như trong training)
        categorical_cols = ['gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'city']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))  # Fit_transform nếu không có encoder lưu sẵn; thực tế nên load encoder nếu có
        
        # Features (tương tự training: điều chỉnh dựa trên schema của bạn)
        features = ['city_development_index', 'training_hours'] + categorical_cols
        
        X = data[features]
        
        return X
    raise ValueError('Failed to prepare inputs')

def get_prediction(input_data, model):
    '''Make predictions using model (binary classification)'''
    prediction = model.predict(input_data)  # Hoặc model.predict_proba(input_data)[:, 1] nếu muốn probability
    return prediction

def return_output(y_pred, content_type):
    '''Format the output'''
    if content_type == 'application/json':
        return json.dumps({'predictions': y_pred.tolist()})  # Trả về list predictions (0/1)
    raise ValueError('Failed to send output')


# Load the model at startup
MODEL_PATH = "/opt/ml/model"  # Đường dẫn mặc định cho SageMaker
model = None

try:
    model = get_model(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {str(e)}")

# Ping endpoint
@app.route("/ping", methods=["GET"])
def ping():
    """SageMaker health check endpoint."""
    if model is None:
        return Response(status=500)
    return Response(status=200)

# Invocation endpoint
@app.route("/invocations", methods=["POST"])
def invocations():
    """SageMaker inference endpoint."""
    if model is None:
        return Response(response="Model not loaded", status=500)

    content_type = flask.request.content_type
    try:
        # Read the request body
        request_body = flask.request.get_data(as_text=True)

        # Prepare input
        X = prepare_input(request_body, content_type)

        # Make predictions
        y_pred = get_prediction(X, model)

        # Format output
        output = return_output(y_pred, content_type)

        return Response(
            response=output,
            status=200,
            mimetype="application/json"
        )
    except ValueError as ve:
        return Response(response=str(ve), status=400)
    except Exception as e:
        return Response(response=f"Prediction error: {str(e)}", status=500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)