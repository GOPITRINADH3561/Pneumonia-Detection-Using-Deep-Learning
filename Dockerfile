FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

ENTRYPOINT ["python3"]
CMD ["app.py", "runserver", "0.0.0.0:8080"]