# syntax=docker/dockerfile:1
# İki hedef (target):
#   dss   : hafif imaj. main.py yalnızca outputs/ okur (DSS, sözleşme kontrolü, demo). Veri gerekmez.
#   train : tam imaj. src/model_*.py ile eğitim (requirements.txt). Veri çalışırken bağlanır.
# Örnek: docker compose run --rm dss    |    docker compose --profile train run --rm train-26

FROM python:3.11.13-slim-bookworm AS dss
LABEL org.opencontainers.image.source="https://github.com/GamzeSerin/bitirme-projesi" \
      org.opencontainers.image.description="IoT güvenliği: çok veri setli DSS"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3"
# Root olmayan kullanıcı (uid 1000: çoğu Linux/Mac kullanıcısıyla aynı, bağlanan klasörlere yazabilir)
RUN groupadd --system --gid 1000 app && useradd --system --uid 1000 --gid app --no-create-home app
COPY --chown=1000:1000 main.py ./
COPY --chown=1000:1000 src ./src
USER app
ENTRYPOINT ["python", "main.py"]

FROM dss AS train
USER root
# lightgbm/xgboost OpenMP çalışma zamanı ister
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install -r /tmp/requirements.txt matplotlib
USER app
ENTRYPOINT []
CMD ["python", "-c", "print('Bir eğitim servisi seçin: train-23 / train-26')"]
