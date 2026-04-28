FROM python:3.12-slim

WORKDIR /app

COPY requirements/requirements-app.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY webapp/ ./webapp/
COPY params.yaml .

EXPOSE 8501 8002

CMD ["streamlit", "run", "webapp/app.py", \
     "--server.fileWatcherType=none", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]