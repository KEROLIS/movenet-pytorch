FROM python:3.8-slim-buster

# Install required packages (for opencv dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*

RUN pip install fastapi uvicorn numpy opencv-python torch torchvision python-multipart

# Clone the GitHub repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/KEROLIS/movenet-pytorch

# Set the working directory
WORKDIR /movenet-pytorch

# Install the project dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

# Load all the models
RUN python -c "import movenet"

# Serve the FastAPI app
EXPOSE 5000
CMD ["uvicorn", "pe_api:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--no-access-log"]
