FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements
COPY backend/ml/requirements.txt requirements.txt
# Install requirements
RUN pip install -r requirements.txt

# FIXME: remove copy data and use a db
COPY data/jb-swisshacks2024/*.csv data/

COPY backend/ml/app app

CMD [ "python3", "-m", "app.main" ]