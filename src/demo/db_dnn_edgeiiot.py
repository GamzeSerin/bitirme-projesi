"""Canlı demo, 4. adım: veritabanı (PostgreSQL; deneme için SQLite). Oynatmanın kararlarını saklar, web arayüzü buradan okur.

Bağlantı adresi .env dosyasındaki DATABASE_URL'den okunur (şablon: .env.example; .env GitHub'a gitmez):
  postgresql+psycopg://<kullanıcı>:<şifre>@localhost:5432/<veritabanı>
  sqlite:///outputs/dnn_edgeiiot/demo/ids_demo.db           (kurulumsuz deneme)

Tablolar:
  runs         bir oynatma: paket klasörü, senaryo (görülmemiş aile), eşikler, tarih
  sources      oynatılan PCAP'ler: dosya, etiket, tür (normal / bilinen / görülmemiş), paket sayısı
  packets      her paketin kararı: NORMAL / SALDIRI / KONTROL / ALARM, tahmin sınıfı, skorlar, eşiğe uzaklık, bağlantı anahtarı
  connections  bağlantı (protokol + sırasız uç noktalar) özeti ve bağlantı kuralının sonucu (flagged, reason)
  alerts       analiste giden bağlantılar (bağlantı kuralı); analistin kararı (status, note) ve açıklama (explanation, JSON metni;
               XAI-2 / XAI-3 sonra doldurulacak)
Bağlantı kuralı (README "Canlı demo, 3. adım"): içerikli (tcp.len > 0) bir paketi NORMAL değilse VEYA bir paketi SALDIRI ise.

Kullanım:
  python src/demo/db_dnn_edgeiiot.py init       tabloları kurar (varsa dokunmaz)
  python src/demo/db_dnn_edgeiiot.py check      tablo başına satır sayısı ve son oynatmalar
  python src/demo/replay_dnn_edgeiiot.py ... --db   oynatmayı veritabanına yazar
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES = ["runs", "sources", "packets", "connections", "alerts"]


# ---------------------------------------------------------------- bağlantı
def load_env(path=None):
    """.env'deki ANAHTAR=değer satırlarını ortam değişkenine yükler (zaten tanımlı olanlara dokunmaz)."""
    path = path or os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def database_url(url=None):
    load_env()
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL yok: .env.example'ı .env adıyla kopyalayıp adresi yaz")
    return url


def safe_url(url):
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", url)


class DB:
    """sqlite3 ve psycopg için ince ortak katman (yer tutucu ? / %s farkını gizler)."""

    def __init__(self, url):
        self.url = url
        if url.startswith("sqlite:///"):
            import sqlite3
            path = url[len("sqlite:///"):]
            if not os.path.isabs(path):
                path = os.path.join(ROOT, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.kind, self.ph = "sqlite", "?"
            self.conn = sqlite3.connect(path)
            self.conn.execute("PRAGMA foreign_keys = ON")
        elif url.startswith(("postgresql", "postgres")):
            try:
                import psycopg
            except ImportError:
                raise SystemExit("psycopg yok: pip install 'psycopg[binary]'")
            self.kind, self.ph = "pg", "%s"
            self.conn = psycopg.connect(re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql://", url))
        else:
            raise SystemExit(f"Desteklenmeyen adres: {safe_url(url)}")

    def q(self, sql):
        return sql.replace("?", self.ph)

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(self.q(sql), params)
        return cur

    def insert(self, table, row):
        cols = list(row)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) RETURNING id"
        return self.execute(sql, [row[c] for c in cols]).fetchone()[0]

    def insert_many(self, table, df):
        cols = list(df.columns)
        rows = [tuple(_py(v) for v in r) for r in df.itertuples(index=False, name=None)]
        sql = self.q(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})")
        cur = self.conn.cursor()
        cur.executemany(sql, rows)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# ---------------------------------------------------------------- şema
def schema(kind):
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if kind == "sqlite" else "BIGSERIAL PRIMARY KEY"
    ts = "TEXT DEFAULT CURRENT_TIMESTAMP" if kind == "sqlite" else "TIMESTAMPTZ DEFAULT now()"
    js = "TEXT" if kind == "sqlite" else "JSONB"
    fk = "INTEGER" if kind == "sqlite" else "BIGINT"
    return [
        f"""CREATE TABLE IF NOT EXISTS runs (
            id {pk}, created_at {ts}, bundle TEXT, scenario TEXT NOT NULL, unseen_family TEXT,
            known_classes {js}, tau_alarm DOUBLE PRECISION, tau_guard DOUBLE PRECISION, max_packets INTEGER,
            n_packets INTEGER, note TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS sources (
            id {pk}, run_id {fk} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            pcap TEXT NOT NULL, label TEXT, family TEXT, kind TEXT, n_packets INTEGER)""",
        f"""CREATE TABLE IF NOT EXISTS packets (
            id {pk}, run_id {fk} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            source_id {fk} NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            t_rel DOUBLE PRECISION, frame INTEGER, src TEXT, dst TEXT, proto TEXT, sport INTEGER, dport INTEGER,
            tcp_len INTEGER, decision TEXT NOT NULL, pred_class TEXT, confidence DOUBLE PRECISION,
            disagree DOUBLE PRECISION, energy DOUBLE PRECISION, alarm_margin DOUBLE PRECISION,
            guard_margin DOUBLE PRECISION, conn_key TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS connections (
            id {pk}, run_id {fk} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            source_id {fk} NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            conn_key TEXT NOT NULL, proto TEXT, end_a TEXT, end_b TEXT, first_t DOUBLE PRECISION,
            last_t DOUBLE PRECISION, n_packets INTEGER, n_content INTEGER, n_normal INTEGER, n_saldiri INTEGER,
            n_kontrol INTEGER, n_alarm INTEGER, top_pred_class TEXT, max_alarm_margin DOUBLE PRECISION,
            flagged BOOLEAN NOT NULL, reason TEXT)""",
        f"""CREATE TABLE IF NOT EXISTS alerts (
            id {pk}, run_id {fk} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            connection_id {fk} NOT NULL REFERENCES connections(id) ON DELETE CASCADE,
            created_at {ts}, t_rel DOUBLE PRECISION, kind TEXT, status TEXT NOT NULL DEFAULT 'acik',
            analyst_note TEXT, explanation {js}, updated_at {ts})""",
        "CREATE INDEX IF NOT EXISTS ix_packets_run ON packets(run_id, t_rel)",
        "CREATE INDEX IF NOT EXISTS ix_conn_run ON connections(run_id, flagged)",
        "CREATE INDEX IF NOT EXISTS ix_alerts_run ON alerts(run_id, status)",
    ]


def init(db):
    for s in schema(db.kind):
        db.execute(s)
    db.commit()


# ---------------------------------------------------------------- bağlantı özeti (bağlantı kuralı)
def connection_table(ev):
    """ev: oynatmanın olay tablosu (source, conn, decision, tcp_len, ...) → kaynak + bağlantı başına bir satır."""
    content = ev["tcp_len"] > 0
    flagged_pkt = ev["decision"] != "NORMAL"
    d = ev.assign(_content=content, _cflag=flagged_pkt & content, _sal=ev["decision"] == "SALDIRI",
                  _nor=ev["decision"] == "NORMAL", _kon=ev["decision"] == "KONTROL", _alm=ev["decision"] == "ALARM")
    g = d.groupby(["source", "conn"], sort=False)
    t = g.agg(proto=("proto", "first"), first_t=("t_rel", "min"), last_t=("t_rel", "max"), n_packets=("decision", "size"),
              n_content=("_content", "sum"), n_normal=("_nor", "sum"), n_saldiri=("_sal", "sum"),
              n_kontrol=("_kon", "sum"), n_alarm=("_alm", "sum"), cflag=("_cflag", "any"), sal=("_sal", "any"),
              max_alarm_margin=("alarm_margin", "max")).reset_index()
    top = (d[d["_sal"]].groupby(["source", "conn"])["pred_class"].agg(lambda s: s.value_counts().index[0])
           .rename("top_pred_class").reset_index())
    t = t.merge(top, on=["source", "conn"], how="left")
    t["flagged"] = t["cflag"] | t["sal"]
    t["reason"] = np.select([t["cflag"] & t["sal"], t["cflag"], t["sal"]],
                            ["icerikli_isaretli+saldiri", "icerikli_isaretli", "saldiri"], default=None)
    parts = t["conn"].str.split("|", n=2, expand=True)
    single = t["conn"].str.startswith("pkt|")          # TCP dışı paket kendi başına birim: uç noktası anahtarda yok
    t["end_a"], t["end_b"] = parts[1].where(~single), parts[2].where(~single)
    return t.drop(columns=["cflag", "sal"]).rename(columns={"conn": "conn_key"})


# ---------------------------------------------------------------- oynatmayı yaz
def save_run(db, ev, scen, bundle, max_packets, kinds, note=None):
    """ev: replay'in olay tablosu (conn sütunuyla). kinds: kaynak → 'normal' / 'bilinen' / 'görülmemiş'. Dönüş: run_id."""
    m = scen.meta
    run_id = db.insert("runs", {"bundle": os.path.abspath(bundle), "scenario": m["unseen_family"],
                                "unseen_family": m["unseen_family"], "known_classes": json.dumps(scen.known),
                                "tau_alarm": float(scen.tau_alarm), "tau_guard": float(scen.tau_guard),
                                "max_packets": int(max_packets), "n_packets": int(len(ev)), "note": note})
    sid = {}
    for (src, lab), g in ev.groupby(["source", "true_label"], sort=False):
        sid[src] = db.insert("sources", {"run_id": run_id, "pcap": src, "label": lab,
                                         "family": g["true_family"].iloc[0], "kind": kinds.get(src),
                                         "n_packets": int(len(g))})
    pk = ev.rename(columns={"conn": "conn_key"})[
        ["t_rel", "frame", "src", "dst", "proto", "sport", "dport", "tcp_len", "decision", "pred_class", "confidence",
         "disagree", "energy", "alarm_margin", "guard_margin", "conn_key"]].copy()
    pk.insert(0, "source_id", ev["source"].map(sid).to_numpy())
    pk.insert(0, "run_id", run_id)
    db.insert_many("packets", pk)

    ct = connection_table(ev)
    ct.insert(0, "source_id", ct.pop("source").map(sid).to_numpy())
    ct.insert(0, "run_id", run_id)
    ct["flagged"] = ct["flagged"].astype(bool)
    cols = ["run_id", "source_id", "conn_key", "proto", "end_a", "end_b", "first_t", "last_t", "n_packets", "n_content",
            "n_normal", "n_saldiri", "n_kontrol", "n_alarm", "top_pred_class", "max_alarm_margin", "flagged", "reason"]
    db.insert_many("connections", ct[cols])
    # analiste gidenler: işaretli bağlantıların her biri bir uyarı (ilk işaretli paketin zamanıyla)
    flagged = db.execute("SELECT id, source_id, conn_key, reason FROM connections WHERE run_id = ? AND flagged = ?",
                         (run_id, True)).fetchall()
    rule = ((ev["decision"] != "NORMAL") & (ev["tcp_len"] > 0)) | (ev["decision"] == "SALDIRI")
    first = ev[rule].groupby(["source", "conn"])["t_rel"].min()
    inv = {v: k for k, v in sid.items()}
    alerts = pd.DataFrame([{"run_id": run_id, "connection_id": cid, "t_rel": float(first[(inv[s], ck)]),
                            "kind": reason, "status": "acik"} for cid, s, ck, reason in flagged],
                          columns=["run_id", "connection_id", "t_rel", "kind", "status"])
    if len(alerts):
        db.insert_many("alerts", alerts)
    db.commit()
    return run_id, len(ct), len(alerts)


def check(db):
    for t in TABLES:
        print(f"  {t:12s} {db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} satır")
    rows = db.execute("SELECT id, created_at, scenario, n_packets FROM runs ORDER BY id DESC LIMIT 5").fetchall()
    for r in rows:
        print(f"  oynatma {r[0]}: {r[1]} senaryo {r[2]}, {r[3]} paket")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init", "check"])
    p.add_argument("--url", default=None, help="Verilmezse .env'deki DATABASE_URL")
    a = p.parse_args()
    url = database_url(a.url)
    db = DB(url)
    print(f"Veritabanı: {safe_url(url)}")
    if a.command == "init":
        init(db)
        print("Tablolar hazır: " + ", ".join(TABLES))
    check(db)
    db.close()


if __name__ == "__main__":
    main()
