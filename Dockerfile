# Use an official Python image
FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy files to the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run both backend and frontend
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501"]
