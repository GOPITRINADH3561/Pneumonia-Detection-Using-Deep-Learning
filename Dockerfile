#Use the official Python image as a base image
FROM python:3.8

# Install dependencies
RUN apt update
RUN apt install python3-pip -y
RUN pip3 install flask==2.1.2
RUN pip3 install scikit-learn==1.0.2
RUN pip3 install tensorflow==2.9.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY lung.h5 /app

# Expose the port on which your Flask app will run
EXPOSE 8080

ENV FLASK_APP = app.py

# Command to run the application
CMD ["python3", "-m", "flask", "run", "--host-0.0.0.0"]