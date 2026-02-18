#!/bin/bash

# Check if conda.yaml exists in the model artifact
if [ -f "/opt/ml/model/conda.yaml" ]; then
    echo "Found conda.yaml, creating Conda environment..."
    # Create a new Conda environment from conda.yaml
    conda env create -f /opt/ml/model/conda.yaml -n model_env
    # Activate the environment
    source /opt/conda/bin/activate model_env
else
    echo "Warning: conda.yaml not found in /opt/ml/model, using base environment"
fi

# Run gunicorn with sagemaker_serve.py
exec gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 60 sagemaker_serve:app