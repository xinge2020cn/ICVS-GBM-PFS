FROM python:3.11-slim

WORKDIR /app
COPY deployment/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY deployment /app/deployment

ENV ICVS_MODEL_ARTIFACT=/model/icvs_model.joblib
EXPOSE 8000
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
