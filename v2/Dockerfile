FROM python:3.13-slim
WORKDIR /app
COPY v2/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
    && useradd --create-home --uid 10001 analyst \
    && mkdir -p /data /app/.streamlit \
    && chown -R analyst:analyst /data /app
COPY app/modules/feature_engineering/feature_service.py /app/app/modules/feature_engineering/feature_service.py
COPY v2 /app/v2
COPY v2/.streamlit/config.toml /app/.streamlit/config.toml
ENV PYTHONUNBUFFERED=1 AUTOINSIGHT_MODE=demo AUTOINSIGHT_DATA_DIR=/data
USER analyst
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -m v2.health
CMD ["python", "-m", "v2.launch"]
