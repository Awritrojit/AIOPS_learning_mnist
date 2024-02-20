# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install required libraries
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "/app/app.py"]
