# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
# Using the correct filename from the project structure
COPY requirements_v3.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_v3.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main_api.py when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "src.main_api:app", "--host", "0.0.0.0", "--port", "8000"]