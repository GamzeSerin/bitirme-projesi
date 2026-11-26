"""Canlı demo, 5. adım (arayüz): kayıtlı bir oynatmadan sahneli sunum zaman çizelgesi ve analist kararları.

Akış: replay_dnn_edgeiiot.py --db oynatmayı veritabanına yazar (runs, sources, packets, connections, alerts). Arayüz bu
kaydı KENDİ saatiyle sahne sahne oynatır (kayıttan oynatma; sunumda modelin hızına bağlı değildir). Kararlar (NORMAL /
SALDIRI / KONTROL / ALARM), bağlantı kuralı ve analiste gidenler veritabanındaki gerçek değerlerdir; arayüz yalnız
nereye/ne zaman çizileceğini seçer.

Kaynakların rolü (sources.kind + dosya adı):
  normal   kind 'normal' ve dosya adı yeni cihaz kalıbına uymuyor (arka plan: bütün sahnelerde akar)
  newdev   kind 'normal' ve dosya adı yeni cihaz kalıbına uyuyor (varsayılan 'modbus'; eğitimde neredeyse hiç yok)
  known    kind 'bilinen'   (modelin eğitimde gördüğü saldırı ailesi)
  unseen   kind 'GÖRÜLMEMİŞ' (senaryonun dışarıda bıraktığı aile)

Sahneler (varsayılan; her biri W saniyelik kayıt penceresi, sunum saatinde arka arkaya):
  1 Normal akış          yalnız normal kaynaklar
  2 Bilinen saldırı      + known
  3 Görülmemiş saldırı   + unseen
  4 Yeni cihaz           + newdev
Normal kaynaklar her sahnede kendi kaydının SONRAKİ penceresinden akar (sahne k: [k·W, (k+1)·W)); özel kaynak kendi
kaydının [0, W) penceresinden. Sunum zamanı = k·W + (t_rel − pencere başı). Bir rolde kaynak yoksa o sahne atlanmaz,
yalnız arka plan akar (arayüz bunu söyler).

Kayma göstergesi (README #14): saniye başına
  alert_rate   son 10 sn'de başlayan bağlantıların analiste giden payı (bağlantı kuralı; SALDIRI otomatik engellenir, sayılmaz)
  novel_share  son 10 sn'deki paketlerin "eğitimde hiç görülmemiş vektör" payı, packets.novel sütunu VARSA;
               yoksa ALARM + KONTROL paketlerinin payı (model şüphesi) ve novel_source = 'model_suphesi'.
Uyarı eşiği (ikisi birden yükselince "ağda yeni bir şey var"): alert_rate > 0,05 ve novel_share > 0,10 (sabit,
sunum için seçildi; deney değildir).

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
FAMILY_OF_CLASS = {"BENIGN": "NORMAL"}  # diğer sınıflar zaten aile adıdır (koşul C ortak taksonomi)


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
    return re.sub(r"[_\s]+", " ", stem).strip()


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
    pcols = ["source_id", "t_rel", "src", "dst", "dport", "tcp_len", "decision", "pred_class", "alarm_margin",
             "guard_margin", "conn_key"] + (["novel"] if novel_col else [])
    pk = _df(db, f"SELECT {', '.join(pcols)} FROM packets WHERE run_id = ?", (run_id,), pcols)
    conn = _df(db, "SELECT id, source_id, conn_key, end_a, end_b, n_packets, n_content, n_saldiri, n_kontrol, n_alarm, "
                   "top_pred_class, max_alarm_margin, flagged, reason FROM connections WHERE run_id = ?", (run_id,),
               ["id", "source_id", "conn_key", "end_a", "end_b", "n_packets", "n_content", "n_saldiri", "n_kontrol",
                "n_alarm", "top_pred_class", "max_alarm_margin", "flagged", "reason"])
    al = _df(db, "SELECT id, connection_id, t_rel, kind, status, analyst_note, explanation FROM alerts WHERE run_id = ?",
             (run_id,), ["id", "connection_id", "t_rel", "kind", "status", "analyst_note", "explanation"])
    role_of = dict(zip(src["id"], src["role"]))
    pk["role"] = pk["source_id"].map(role_of)

    # ---- sahnelere böl
    parts, scenes = [], []
    for k, (key, name, special) in enumerate(SCENES):
        t0 = k * window
        bg = pk[(pk["role"] == "normal") & (pk["t_rel"] >= t0) & (pk["t_rel"] < t0 + window)].assign(t=lambda d: d["t_rel"] - t0)
        sp = (pk[(pk["role"] == special) & (pk["t_rel"] < window)].assign(t=lambda d: d["t_rel"])
              if special else pk.iloc[0:0].assign(t=[]))
        chunk = pd.concat([bg, sp])
        # çok yoğun kaynakları seyrelt (yalnız çizim; sayaçlar tam sayıdan gelir): NORMAL dışı kararlar hep kalır
        keep = []
        for sid, g in chunk.groupby("source_id"):
            imp = g[g["decision"] != "NORMAL"]
            rest = g[g["decision"] == "NORMAL"]
            room = max(max_per_source_scene - len(imp), 0)
            if len(rest) > room:
                rest = rest.iloc[np.linspace(0, len(rest) - 1, room).astype(int)] if room else rest.iloc[0:0]
            keep.append(pd.concat([imp, rest]))
        drawn = pd.concat(keep) if keep else chunk
        counts = chunk["decision"].value_counts().to_dict()
        scenes.append({"key": key, "name": name, "start": t0, "end": t0 + window, "has_special": bool(special is None or len(sp)),
                       "counts": {d: int(counts.get(d, 0)) for d in ("NORMAL", "SALDIRI", "KONTROL", "ALARM")},
                       "sources": sorted(set(src.set_index("id").loc[chunk["source_id"].unique(), "pcap"]))
                       if len(chunk) else []})
        parts.append(drawn.assign(t=drawn["t"] + t0, scene=k, _seg=k))
    tl = pd.concat(parts).sort_values("t", kind="stable").reset_index(drop=True)

    # ---- bağlantılar: sahnede ilk paketi görülen bağlantılar, sunum zamanıyla
    first = (tl.groupby(["scene", "source_id", "conn_key"])["t"].min().rename("t").reset_index())
    cj = first.merge(conn, on=["source_id", "conn_key"], how="left")
    alerts_by_conn = al.set_index("connection_id")
    def outcome(r):
        if r["n_saldiri"] and r["n_saldiri"] > 0:
            return "SALDIRI"
        if r["flagged"]:
            return "ALARM" if (r["n_alarm"] or 0) > 0 else "KONTROL"
        return "NORMAL"
    cj["outcome"] = cj.apply(outcome, axis=1)
    cj["role"] = cj["source_id"].map(role_of)
    # uyarı zamanı: sahnedeki ilk işaretli içerikli paket (yoksa ilk SALDIRI paketi), yoksa bağlantının başlangıcı
    rule = ((tl["decision"] != "NORMAL") & (tl["tcp_len"] > 0)) | (tl["decision"] == "SALDIRI")
    ft = tl[rule].groupby(["scene", "source_id", "conn_key"])["t"].min().rename("t_alert").reset_index()
    cj = cj.merge(ft, on=["scene", "source_id", "conn_key"], how="left")
    connections, alerts = [], []
    for r in cj.itertuples(index=False):
        c = {"id": int(r.id) if pd.notna(r.id) else None, "t": round(float(r.t), 3), "scene": int(r.scene),
             "role": r.role, "src": r.end_a, "dst": r.end_b, "outcome": r.outcome, "flagged": bool(r.flagged),
             "pred": r.top_pred_class if isinstance(r.top_pred_class, str) else None}
        connections.append(c)
        if bool(r.flagged) and pd.notna(r.id) and int(r.id) in alerts_by_conn.index:
            a = alerts_by_conn.loc[int(r.id)]
            a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
            expl = a["explanation"]
            alerts.append({"id": int(a["id"]), "connection_id": int(r.id), "scene": int(r.scene), "role": r.role,
                           "t": round(float(r.t_alert if pd.notna(r.t_alert) else r.t), 3),
                           "kind": a["kind"], "status": a["status"], "note": a["analyst_note"],
                           "explanation": expl if isinstance(expl, (dict, list)) else (json.loads(expl) if expl else None),
                           "src": r.end_a, "dst": r.end_b, "outcome": r.outcome, "pred": c["pred"],
                           "n_packets": int(r.n_packets), "n_content": int(r.n_content), "n_saldiri": int(r.n_saldiri),
                           "n_kontrol": int(r.n_kontrol), "n_alarm": int(r.n_alarm),
                           "max_alarm_margin": None if pd.isna(r.max_alarm_margin) else round(float(r.max_alarm_margin), 2)})
    connections.sort(key=lambda c: c["t"])
    alerts.sort(key=lambda a: a["t"])
    all_conn = list(connections)
    # sahne sayaçları TAM sayılardan; çizim için bağlantı listesi seyreltilir (analiste gidenler hep kalır)
    for k, sc in enumerate(scenes):
        cs = [c for c in connections if c["scene"] == k]
        sc["conn"] = {"total": len(cs), "blocked": sum(c["outcome"] == "SALDIRI" for c in cs),
                      "to_analyst": sum(c["flagged"] and c["outcome"] != "SALDIRI" for c in cs)}
    imp = [c for c in connections if c["flagged"] and c["outcome"] != "SALDIRI"]
    rest = [c for c in connections if not (c["flagged"] and c["outcome"] != "SALDIRI")]
    if len(rest) > max_connections:
        rest = [rest[i] for i in np.linspace(0, len(rest) - 1, max_connections).astype(int)]
    connections = sorted(imp + rest, key=lambda c: c["t"])
    # analist masasına yalnız tanınmayanlar gelir; bilinen saldırı (SALDIRI) otomatik engellenir, sayısı sahnede
    alerts = [a for a in alerts if a["outcome"] != "SALDIRI"]

    # ---- kayma göstergesi (saniyelik)
    total = len(SCENES) * window
    secs = np.arange(0, int(np.ceil(total)) + 1)
    ct = np.array([c["t"] for c in all_conn])
    cf = np.array([c["flagged"] and c["outcome"] != "SALDIRI" for c in all_conn], bool)  # analiste giden
    full = []  # novel payı için seyreltilmemiş paketler gerekir: sahne parçalarından yeniden say
    for k, (key, name, special) in enumerate(SCENES):
        t0 = k * window
        bg = pk[(pk["role"] == "normal") & (pk["t_rel"] >= t0) & (pk["t_rel"] < t0 + window)]
        sp = pk[(pk["role"] == special) & (pk["t_rel"] < window)] if special else pk.iloc[0:0]
        tt = np.concatenate([bg["t_rel"].to_numpy() - t0, sp["t_rel"].to_numpy()]) + t0
        nv = (np.concatenate([bg["novel"].to_numpy(bool), sp["novel"].to_numpy(bool)]) if novel_col else
              np.isin(np.concatenate([bg["decision"].to_numpy(), sp["decision"].to_numpy()]), ["ALARM", "KONTROL"]))
        full.append((tt, nv))
    pt = np.concatenate([f[0] for f in full]); pn = np.concatenate([f[1] for f in full])
    drift = []
    for s in secs:
        cm = (ct > s - DRIFT_WIN) & (ct <= s); pm = (pt > s - DRIFT_WIN) & (pt <= s)
        ar = float(cf[cm].mean()) if cm.any() else 0.0
        ns = float(pn[pm].mean()) if pm.any() else 0.0
        drift.append({"t": int(s), "alert_rate": round(ar, 4), "novel_share": round(ns, 4),
                      "warn": bool(ar > ALERT_HI and ns > NOVEL_HI)})

    devices = [{"source_id": int(r.id), "name": device_name(r.pcap), "role": r.role, "label": r.label}
               for r in src.itertuples(index=False)]
    cols = ["t", "source_id", "decision", "pred_class", "alarm_margin", "guard_margin", "tcp_len", "scene"]
    packets = {c: (tl[c].round(3).tolist() if tl[c].dtype.kind == "f" else tl[c].tolist()) for c in cols}
    packets["source_id"] = [int(x) for x in packets["source_id"]]
    packets["tcp_len"] = [int(x) for x in packets["tcp_len"]]
    packets["scene"] = [int(x) for x in packets["scene"]]
    return _clean({"run": {"id": run[0], "scenario": run[1], "unseen_family": run[2], "known_classes": known,
                    "tau_alarm": run[4], "tau_guard": run[5], "n_packets": run[6]},
            "window": window, "duration": total, "scenes": scenes, "devices": devices,
            "packets": packets, "connections": connections, "alerts": alerts,
            "drift": {"series": drift, "window": DRIFT_WIN, "alert_hi": ALERT_HI, "novel_hi": NOVEL_HI,
                      "novel_source": "novel" if novel_col else "model_suphesi"},
            "family_of_class": {c: FAMILY_OF_CLASS.get(c, c) for c in known}})


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
