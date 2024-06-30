FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements.txt
# Install requirements
RUN pip install -r requirements.txt

COPY app app

CMD [ "python3", "-m", "app.main" ]