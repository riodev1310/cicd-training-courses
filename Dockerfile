FROM ghcr.io/mlflow/mlflow:v2.21.3

# Install system dependencies required by SageMaker
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget curl nginx ca-certificates bzip2 build-essential cmake git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for Conda environment management
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
&& /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
&& rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Install AWS CLI 
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& ./aws/install \
&& rm -rf awscliv2.zip aws

# Install AWS SDKs, gunicorn, and other dependencies
RUN pip install boto3==1.37.30 gunicorn==23.0.0 flask==2.2.5 pandas==2.2.3 numpy==2.2.3

# Copy the SageMaker serving script and entrypoint script
COPY sagemaker_serve.py /sagemaker_serve.py
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint.sh executable
RUN chmod +x /entrypoint.sh


# Expose port 8080 for SageMaker
EXPOSE 8080

# Set SageMaker environment variables
ENV SAGEMAKER_PROGRAM=sagemaker_serve.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model


# Use entrypoint.sh to activate Conda environment and run gunicorn
ENTRYPOINT ["/entrypoint.sh"]