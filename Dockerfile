FROM python:3.8-slim-buster

# Install required packages
RUN pip install fastapi uvicorn numpy opencv-python torch torchvision

# Clone the GitHub repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/KEROLIS/movenet-pytorch

# Set the working directory
WORKDIR /movenet-pytorch

# Install the project dependencies
RUN pip install -r requirements.txt

# Copy the project files
COPY . .

# Load all the models
RUN python -c "import movenet"

# Serve the FastAPI app
EXPOSE 5000
CMD ["uvicorn", "pe_api:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--no-access-log"]
