# Use an official PyTorch runtime with CUDA as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD /app /app
ADD /legal_eval/spacy/output/model-best /legal_eval/spacy/output/model-best
ADD /legal_eval/spacy/gpu-RoBERTa/output/model-best /legal_eval/spacy/gpu-RoBERTa/output/model-best
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt


# Make port 5000 available for the app
EXPOSE 8000

# Run app.py when the container launches
CMD ["gunicorn",  "-b", "0.0.0.0:8000", "wsgi:app"]
