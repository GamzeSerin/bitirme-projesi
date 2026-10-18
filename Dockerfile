# İki hedef (target):
#   dss   : hafif imaj. main.py yalnızca outputs/ okur (DSS, sözleşme kontrolü, demo). Veri gerekmez.
#   train : tam imaj. src/model_*.py ile eğitim (requirements.txt). Veri dışarıdan bağlanır.
# Örnek: docker compose up dss    |    docker compose --profile train run --rm train-26

FROM python:3.11-slim AS dss
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
RUN pip install --no-cache-dir "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3"
COPY main.py ./
COPY src ./src
ENTRYPOINT ["python", "main.py"]

FROM dss AS train
# lightgbm/xgboost OpenMP çalışma zamanı ister
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt matplotlib
ENTRYPOINT []
CMD ["python", "-c", "print('Bir eğitim servisi seçin: train-23 / train-26')"]
