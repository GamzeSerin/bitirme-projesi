"""Canlı demo, 5. adım: web arayüzü sunucusu (FastAPI). Veritabanından okur, sahneli zaman çizelgesini tarayıcıya verir.

Çalıştırma (yerel bilgisayar; .env'de DATABASE_URL, önce replay ... --db ile en az bir oynatma kaydedilmiş olmalı):
  pip install -r src/demo/requirements-demo.txt
  uvicorn src.demo.web.app_dnn_edgeiiot:app --port 8000      (depo kökünden)
  ya da: python src/demo/web/app_dnn_edgeiiot.py --port 8000
  tarayıcı: http://localhost:8000          (?run=<id> ile belirli bir oynatma; verilmezse en sonuncusu)

Uç noktalar:
  GET  /api/runs                         kayıtlı oynatmalar (en yeni önce)
  GET  /api/runs/{id}/timeline           sahneli zaman çizelgesi (timeline_dnn_edgeiiot.timeline); ?window=30&newdev=modbus
  POST /api/alerts/{id}                  analist kararı {"status": ..., "note": ...} → alerts tablosu
Kurallar timeline_dnn_edgeiiot.py'nin docstring'inde (sahneler, kayma göstergesi, durum değerleri). Sunucu yalnız
localhost'ta dinler (varsayılan 127.0.0.1); veritabanı adresi ve şifresi tarayıcıya gönderilmez.
"""

import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import db_dnn_edgeiiot as dbm  # noqa: E402
import timeline_dnn_edgeiiot as tlm  # noqa: E402

app = FastAPI(title="Bekçi demo")
_lock = threading.Lock()   # tek bağlantı; istekler sırayla
_db = None


def db():
    global _db
    if _db is None:
        _db = dbm.DB(dbm.database_url())
    return _db


class Verdict(BaseModel):
    status: str
    note: str | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/runs")
def runs():
    with _lock:
        return tlm.list_runs(db())


@app.get("/api/runs/{run_id}/timeline")
def timeline(run_id: int, window: float = 30.0, newdev: str = "modbus"):
    with _lock:
        try:
            return tlm.timeline(db(), run_id, window=window, newdev_pattern=newdev)
        except KeyError:
            raise HTTPException(404, f"oynatma yok: {run_id}")


@app.post("/api/alerts/{alert_id}")
def verdict(alert_id: int, v: Verdict):
    with _lock:
        try:
            return tlm.set_alert(db(), alert_id, v.status, v.note)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except KeyError:
            raise HTTPException(404, f"uyarı yok: {alert_id}")


if __name__ == "__main__":
    import argparse

    import uvicorn
    p = argparse.ArgumentParser(description="Bekçi demo arayüzü")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"Veritabanı: {dbm.safe_url(dbm.database_url())}\nArayüz: http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)
