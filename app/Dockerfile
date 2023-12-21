# Use the python 3.10.13-bookworm container image
from python:3.10.13-bookworm

# Set the working directory to /app
WORKDIR /app

# Copy current directory contents into container at /app
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8501

# Run Gunicorn command to start the application
#CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", ":8501", "main:app"]
#CMD ["gunicorn", "-w", "4", "-b", ":8501", "wsgi:main"]
#ENTRYPOINT ["streamlit", "run", "wsgi.py", "--server.enableWebsockets=false", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "wsgi.py"]
#"--server.port=8501", "--server.address=0.0.0.0"]

