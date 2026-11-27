"""Canlı demo, 5. adım (arayüz): kayıtlı bir oynatmadan sahneli sunum zaman çizelgesi ve analist kararları.

Akış: replay_dnn_edgeiiot.py --db oynatmayı veritabanına yazar (runs, sources, packets, connections, alerts). Arayüz bu
kaydı KENDİ saatiyle sahne sahne oynatır (kayıttan oynatma; sunumda modelin hızına bağlı değildir). Kararlar (NORMAL /
SALDIRI / KONTROL / ALARM) ve bağlantılar veritabanındaki gerçek değerlerdir; arayüz yalnız nereye/ne zaman çizileceğini seçer.

Kaynakların rolü (sources.kind + dosya adı):
  normal   kind 'normal' ve dosya adı yeni cihaz kalıbına uymuyor (arka plan: bütün sahnelerde akar)
  newdev   kind 'normal' ve dosya adı yeni cihaz kalıbına uyuyor (varsayılan 'modbus'; eğitimde neredeyse hiç yok)
  known    kind 'bilinen'   (modelin eğitimde gördüğü saldırı ailesi)
  unseen   kind 'GÖRÜLMEMİŞ' (senaryonun dışarıda bıraktığı aile)

Sahneler (her biri W sunum saniyesi, arka arkaya): 1 Normal akış (yalnız normal), 2 Bilinen saldırı (+known),
3 Görülmemiş saldırı (+unseen), 4 Yeni cihaz (+newdev).
ZAMAN SIKIŞTIRMA (27.11.2026, ilk gerçek oynatmadan sonra): PCAP'lerin kayıt süreleri çok farklı (20 bin paket: Water_Level
29 sn, Temperature 20 dk, Modbus 16 gün). Bu yüzden zaman penceresi değil PAKET SIRASI kullanılır: her normal kaynağın paketleri
kayıt sırasıyla 4 eşit parçaya bölünür, sahne k k'inci parçayı oynatır; özel kaynağın bütün paketleri kendi sahnesinde oynar.
Bir parçanın i'nci paketi sahnenin başından W·i/n saniye sonra gösterilir (kayıt zamanı sıkıştırılır, sıra ve kararlar aynen
korunur). Sahne bilgisine her kaynağın gerçek kayıt süresi yazılır; arayüz "kayıt zamanı sıkıştırıldı" der.

Bağlantının sonucu (gösterim; bağlantı kuralı ve kararlar değişmez):
  bilinmeyen (analist kuyruğu)  içerik taşıyan (tcp.len > 0) en az bir paketi ALARM ya da KONTROL. Model "tanımıyorum / emin
                                değilim" dedi. Aynı bağlantıda SALDIRI paketi de varsa ayrıca "engellendi" (blocked) işaretlenir.
  SALDIRI (engellendi)          SALDIRI paketi var, içerikli ALARM/KONTROL yok: model bildiği bir saldırı sanıp kendisi durdurdu.
  NORMAL                        ikisi de yok.
Neden ayrı: görülmemiş saldırıda (WEB) paketlerin bir kısmını model bilinen bir aileye benzetiyor (ilk oynatmada SQL_injection
paketlerinin %68'i SALDIRI, %32'si ALARM). Eski gösterim "SALDIRI varsa engellendi" diyerek bu bağlantıların hepsini analistten
saklıyordu; doğrusu ikisini birlikte göstermek.

Kayma göstergesi (README #14): saniye başına (sunum zamanı)
  alert_rate   son 10 sn'de başlayan bağlantıların analist kuyruğuna giden payı
  novel_share  son 10 sn'deki paketlerin "eğitimde hiç görülmemiş vektör" payı, packets.novel sütunu VARSA;
               yoksa ALARM + KONTROL paketlerinin payı (model şüphesi) ve novel_source = 'model_suphesi'.
Uyarı eşiği (ikisi birden): alert_rate > 0,05 ve novel_share > 0,10 (sabit, sunum için seçildi; deney değildir).

Analist kararı: alerts.status ('acik' → 'saldiri_onay' / 'yanlis_alarm' / 'cihaz_tanit' / 'engelle'), analyst_note.
"""

import json
import re

import numpy as np
import pandas as pd

SCENES = [("normal", "Normal akış", None), ("known", "Bilinen saldırı", "known"),
          ("unseen", "Görülmemiş saldırı", "unseen"), ("newdev", "Yeni cihaz", "newdev")]
STATUSES = {"acik", "saldiri_onay", "yanlis_alarm", "cihaz_tanit", "engelle"}
DRIFT_WIN, ALERT_HI, NOVEL_HI = 10.0, 0.05, 0.10


def _df(db, sql, params, cols):
    return pd.DataFrame(db.execute(sql, params).fetchall(), columns=cols)


def list_runs(db):
    rows = db.execute("SELECT id, created_at, scenario, n_packets, note FROM runs ORDER BY id DESC").fetchall()
    return [{"id": r[0], "created_at": str(r[1]), "scenario": r[2], "n_packets": r[3], "note": r[4]} for r in rows]


def has_column(db, table, col):
    try:
        db.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchall()
        return True
    except Exception:
        db.conn.rollback()
        return False


def roles(sources, newdev_pattern):
    pat = re.compile(newdev_pattern, re.I)

    def role(r):
        k = str(r["kind"] or "")
        if k.lower() == "normal":
            return "newdev" if pat.search(str(r["pcap"])) else "normal"
        if k.lower() == "bilinen":
            return "known"
        # replay 'GÖRÜLMEMİŞ' yazar; büyük İ'nin lower() hali i + birleşik nokta olduğundan kök üzerinden bakılır
        return "unseen" if "RÜLMEM" in k.upper() or "RULMEM" in k.upper() else "other"
    return sources.assign(role=sources.apply(role, axis=1))


def device_name(pcap):
    stem = re.sub(r"\.pcap(ng)?$", "", str(pcap), flags=re.I)
    stem = re.sub(r"^(normal|attack)[_\s-]+", "", stem, flags=re.I)   # dosya adı ön eki (Normal_Water_Level)
    return re.sub(r"[_\s]+", " ", stem).strip()


def _place(pk, src, window):
    """Her pakete sahne ve sunum zamanı ver (paket sırasıyla sıkıştırma; docstring)."""
    pk = pk.sort_values(["source_id", "t_rel"], kind="stable")
    pk["scene"], pk["t"] = -1, np.nan
    role_of = dict(zip(src["id"], src["role"]))
    for sid, g in pk.groupby("source_id", sort=False):
        role, n = role_of.get(sid), len(g)
        if role == "normal":
            part = np.minimum((np.arange(n) * len(SCENES)) // max(n, 1), len(SCENES) - 1)
            start = np.searchsorted(part, part, side="left")
            size = np.bincount(part, minlength=len(SCENES))[part]
            scene, frac = part, (np.arange(n) - start) / np.maximum(size, 1)
        else:
            k = next((i for i, s in enumerate(SCENES) if s[2] == role), None)
            if k is None:
                continue
            scene, frac = np.full(n, k), np.arange(n) / max(n, 1)
        pk.loc[g.index, "scene"] = scene
        pk.loc[g.index, "t"] = scene * window + frac * window
    return pk[pk["scene"] >= 0].sort_values("t", kind="stable").reset_index(drop=True)


def timeline(db, run_id, window=30.0, newdev_pattern="modbus", max_per_source_scene=3000, max_connections=3000):
    """Sunum zaman çizelgesi (JSON'a hazır sözlük)."""
    run = db.execute("SELECT id, scenario, unseen_family, known_classes, tau_alarm, tau_guard, n_packets "
                     "FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)
    known = run[3] if isinstance(run[3], list) else json.loads(run[3] or "[]")
    src = roles(_df(db, "SELECT id, pcap, label, family, kind, n_packets FROM sources WHERE run_id = ?", (run_id,),
                    ["id", "pcap", "label", "family", "kind", "n_packets"]), newdev_pattern)
    novel_col = has_column(db, "packets", "novel")
    pcols = ["source_id", "t_rel", "decision", "pred_class", "alarm_margin", "guard_margin", "tcp_len",
             "conn_key"] + (["novel"] if novel_col else [])
    pk = _df(db, f"SELECT {', '.join(pcols)} FROM packets WHERE run_id = ?", (run_id,), pcols)
    conn = _df(db, "SELECT id, source_id, conn_key, end_a, end_b, n_packets, n_content, n_saldiri, n_kontrol, n_alarm, "
                   "top_pred_class, max_alarm_margin, flagged, reason FROM connections WHERE run_id = ?", (run_id,),
               ["id", "source_id", "conn_key", "end_a", "end_b", "n_packets", "n_content", "n_saldiri", "n_kontrol",
                "n_alarm", "top_pred_class", "max_alarm_margin", "flagged", "reason"])
    al = _df(db, "SELECT id, connection_id, kind, status, analyst_note, explanation FROM alerts WHERE run_id = ?",
             (run_id,), ["id", "connection_id", "kind", "status", "analyst_note", "explanation"])
    role_of = dict(zip(src["id"], src["role"]))
    span = pk.groupby("source_id")["t_rel"].agg(lambda s: float(s.max() - s.min()))
    pk = _place(pk, src, window)
    pk["role"] = pk["source_id"].map(role_of)

    # ---- bağlantı sonucu paketlerden (docstring): içerikli ALARM/KONTROL → bilinmeyen; SALDIRI → engellendi
    pk["_unk"] = (pk["tcp_len"] > 0) & pk["decision"].isin(["ALARM", "KONTROL"])
    pk["_alm"] = (pk["tcp_len"] > 0) & (pk["decision"] == "ALARM")
    pk["_sal"] = pk["decision"] == "SALDIRI"
    key = ["scene", "source_id", "conn_key"]
    cs = pk.groupby(key).agg(t=("t", "min"), unk=("_unk", "any"), alm=("_alm", "any"), sal=("_sal", "any")).reset_index()
    t_unk = pk[pk["_unk"]].groupby(key)["t"].min().rename("t_alert").reset_index()
    cs = cs.merge(t_unk, on=key, how="left").merge(conn, on=["source_id", "conn_key"], how="left")
    cs["role"] = cs["source_id"].map(role_of)
    cs["outcome"] = np.where(cs["unk"], np.where(cs["alm"], "ALARM", "KONTROL"), np.where(cs["sal"], "SALDIRI", "NORMAL"))
    alert_of = {int(r.connection_id): r for r in al.itertuples(index=False)}

    connections, alerts = [], []
    for r in cs.itertuples(index=False):
        cid = int(r.id) if pd.notna(r.id) else None
        c = {"id": cid, "t": round(float(r.t), 3), "scene": int(r.scene), "source_id": int(r.source_id), "role": r.role,
             "src": r.end_a, "dst": r.end_b,
             "outcome": r.outcome, "blocked": bool(r.sal), "to_analyst": bool(r.unk),
             "pred": r.top_pred_class if isinstance(r.top_pred_class, str) else None}
        connections.append(c)
        a = alert_of.get(cid) if cid is not None else None
        if r.unk and a is not None:
            expl = a.explanation
            alerts.append({"id": int(a.id), "connection_id": cid, "scene": int(r.scene), "source_id": int(r.source_id),
                           "role": r.role,
                           "t": round(float(r.t_alert), 3), "kind": a.kind, "status": a.status, "note": a.analyst_note,
                           "explanation": expl if isinstance(expl, (dict, list)) else (json.loads(expl) if expl else None),
                           "src": r.end_a, "dst": r.end_b, "outcome": r.outcome, "blocked": bool(r.sal), "pred": c["pred"],
                           "n_packets": int(r.n_packets), "n_content": int(r.n_content), "n_saldiri": int(r.n_saldiri),
                           "n_kontrol": int(r.n_kontrol), "n_alarm": int(r.n_alarm),
                           "max_alarm_margin": None if pd.isna(r.max_alarm_margin) else round(float(r.max_alarm_margin), 2)})
    connections.sort(key=lambda c: c["t"])
    alerts.sort(key=lambda a: a["t"])
    all_conn = list(connections)

    # ---- sahneler (sayılar seyreltilmeden önce, TAM)
    scenes = []
    for k, (key_, name, special) in enumerate(SCENES):
        chunk = pk[pk["scene"] == k]
        cnt = chunk["decision"].value_counts().to_dict()
        cc = [c for c in all_conn if c["scene"] == k]
        srcs = []
        for sid, g in chunk.groupby("source_id"):
            row = src.set_index("id").loc[sid]
            srcs.append({"source_id": int(sid), "name": device_name(row["pcap"]), "role": role_of[sid],
                         "n_packets": int(len(g)), "record_seconds": round(float(span.get(sid, 0.0)), 1),
                         "counts": {d: int((g["decision"] == d).sum()) for d in ("NORMAL", "SALDIRI", "KONTROL", "ALARM")}})
        scenes.append({"key": key_, "name": name, "start": k * window, "end": (k + 1) * window,
                       "has_special": bool(special is None or (pk["role"] == special).any()),
                       "counts": {d: int(cnt.get(d, 0)) for d in ("NORMAL", "SALDIRI", "KONTROL", "ALARM")},
                       "conn": {"total": len(cc), "blocked": sum(c["blocked"] for c in cc),
                                "to_analyst": sum(c["to_analyst"] for c in cc),
                                "blocked_and_analyst": sum(c["blocked"] and c["to_analyst"] for c in cc)},
                       "sources": srcs})

    # ---- kayma göstergesi (saniyelik, tam paketlerden)
    total = len(SCENES) * window
    ct = np.array([c["t"] for c in all_conn])
    cf = np.array([c["to_analyst"] for c in all_conn], bool)
    pt = pk["t"].to_numpy()
    pn = pk["novel"].to_numpy(bool) if novel_col else pk["decision"].isin(["ALARM", "KONTROL"]).to_numpy()
    drift = []
    for s in np.arange(0, int(np.ceil(total)) + 1):
        cm = (ct > s - DRIFT_WIN) & (ct <= s)
        lo, hi = np.searchsorted(pt, s - DRIFT_WIN, side="right"), np.searchsorted(pt, s, side="right")
        ar = float(cf[cm].mean()) if cm.any() else 0.0
        ns = float(pn[lo:hi].mean()) if hi > lo else 0.0
        drift.append({"t": int(s), "alert_rate": round(ar, 4), "novel_share": round(ns, 4),
                      "warn": bool(ar > ALERT_HI and ns > NOVEL_HI)})

    # ---- çizim için seyreltme (sayaçlar yukarıdaki tam sayılardan gelir): NORMAL dışı kararlar hep kalır
    keep = []
    for _, g in pk.groupby(["scene", "source_id"]):
        imp, rest = g[g["decision"] != "NORMAL"], g[g["decision"] == "NORMAL"]
        room = max(max_per_source_scene - len(imp), 0)
        if len(rest) > room:
            rest = rest.iloc[np.linspace(0, len(rest) - 1, room).astype(int)] if room else rest.iloc[0:0]
        keep.append(pd.concat([imp, rest]))
    tl = pd.concat(keep).sort_values("t", kind="stable")
    imp_c = [c for c in all_conn if c["to_analyst"]]
    rest_c = [c for c in all_conn if not c["to_analyst"]]
    if len(rest_c) > max_connections:
        rest_c = [rest_c[i] for i in np.linspace(0, len(rest_c) - 1, max_connections).astype(int)]
    connections = sorted(imp_c + rest_c, key=lambda c: c["t"])

    devices = [{"source_id": int(r.id), "name": device_name(r.pcap), "role": r.role, "label": r.label,
                "record_seconds": round(float(span.get(r.id, 0.0)), 1)} for r in src.itertuples(index=False)]
    cols = ["t", "source_id", "decision", "pred_class", "alarm_margin", "guard_margin", "tcp_len", "scene"]
    packets = {c: tl[c].round(3).tolist() if tl[c].dtype.kind == "f" else tl[c].tolist() for c in cols}
    for c in ("source_id", "tcp_len", "scene"):
        packets[c] = [int(x) for x in packets[c]]
    return _clean({"run": {"id": run[0], "scenario": run[1], "unseen_family": run[2], "known_classes": known,
                           "tau_alarm": run[4], "tau_guard": run[5], "n_packets": run[6]},
                   "window": window, "duration": total, "time_compressed": True, "scenes": scenes, "devices": devices,
                   "packets": packets, "connections": connections, "alerts": alerts,
                   "drift": {"series": drift, "window": DRIFT_WIN, "alert_hi": ALERT_HI, "novel_hi": NOVEL_HI,
                             "novel_source": "novel" if novel_col else "model_suphesi"}})


def _clean(o):
    """JSON'da NaN/sonsuz geçersizdir (tarayıcı ve FastAPI reddeder): None'a çevir; numpy türlerini Python'a."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def set_alert(db, alert_id, status, note=None):
    if status not in STATUSES:
        raise ValueError(f"geçersiz durum: {status} (olabilecekler: {sorted(STATUSES)})")
    cur = db.execute("UPDATE alerts SET status = ?, analyst_note = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                     (status, note, alert_id))
    db.commit()
    if getattr(cur, "rowcount", 1) == 0:
        raise KeyError(alert_id)
    return {"id": alert_id, "status": status, "note": note}
