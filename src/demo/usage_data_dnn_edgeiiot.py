"""Canlı demo: "kullanım modeli" verisi (Edge-IIoTset). YALNIZ demo modeli için; tezin deney sayıları bunu KULLANMAZ.

Neden (29.10.2026, demo teşhisi): yerelde gerçek normal trafik (Temperature_and_Humidity) oynatılınca paketlerin %51'i ALARM aldı.
En sık paket tipleri (MQTT sunucusuna boş ACK %24,9, SYN, CONNECT, PUBLISH, DISCONNECT) asıl deneyin eğitim verisinde neredeyse
yok (40 bin/etiket örneklemde boş ACK 1 satır, diğerleri 0; temizlenmiş verinin tamamında 1–6 satır): tekrar silme binlerce
kopyayı teke indiriyor, kalan kopya da büyük olasılıkla etiket çakışması sayılıp atılıyor. Açık küme dedektörü bu, ağın en sık
gördüğü normal paketleri doğru biçimde "hiç görmedim" sayıyor. Ayrıca koşul C'de (seq/ack yok) etiketlerin büyük kısmı başka bir
etiketle birebir aynı vektöre düşüyor (Backdoor %89, XSS %79, Password %60, Normal %11; `outputs/dnn_edgeiiot/demo/label_noise_check.json`).

Kural (kullanım modeli):
  1) Temizlik modelin gördüğü özelliklerle (koşul C, 32 özellik) yapılır; hizasız satırlar ve eksik değerler yine atılır.
  2) Tekrarlar SİLİNMEZ: her ince etiketten ham satırlar rastgele çekilir (en fazla cap); sık paket tipleri örneklemde de sıktır.
  3) Çakışmalar ATILMAZ: aynı 32 özellikli vektör birden çok ince etiketle geçiyorsa ham verideki en sık etiketini alır (çoğunluk).
     Bedeli: ayırt edilemeyen saldırı paketleri (ör. saldırı oturumunun boş ACK'i) normal öğrenilir; bunları paket düzeyinde
     hiçbir dedektör ayıramıyordu (ikiz kontrolü). Saldırıyı ele veren içerikli paketler etiketini korur.
  4) Örneklem çoğunluk etiketine göre yapılır (yeniden etiketlenen satır yeni etiketinin havuzuna girer).
Sınırlılık: tekrarlar silinmediği için aynı vektörler hem eğitimde hem testte olur; bu verideki test sayıları başarı ölçüsü
DEĞİLDİR. Demodaki normal trafiğin kopyaları eğitimde bulunabilir (sahada sistem kendi ağının normal trafiğiyle eğitilir);
görülmemiş aile ise eğitimde hiç yoktur, o taraf dürüsttür.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

MANIFEST = "outputs/dnn_edgeiiot/manifest.json"
DROP_C = ["tcp.seq", "tcp.ack"]


def load_usage_data(data_file, cap, seed, drop_conflicts=True):
    """registry loader imzası: X (34 özellik, float32), fine, category, stats. drop_conflicts yok sayılır (çoğunluk kuralı)."""
    from common.taxonomy import edgeiiot_category
    from model_dnn_edgeiiot import LABEL, misaligned_mask
    feats = json.load(open(MANIFEST, encoding="utf-8"))["features"]
    print(f"Okunuyor (kullanım verisi): {data_file}")
    df = pd.read_csv(data_file, low_memory=False, usecols=lambda c: c in feats or c in (LABEL, "frame.time"))
    stats = {"rule": "kullanım modeli: koşul C'de çoğunluk etiketi, tekrarlar korunur", "rows_raw": int(len(df))}
    bad = misaligned_mask(df)
    df = df[~bad]
    stats["rows_misaligned_dropped"] = int(bad.sum())
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).to_numpy()
    X, lab = X[ok].reset_index(drop=True), df[LABEL].astype(str).str.strip().str.upper().to_numpy()[ok]
    stats["rows_after_nan_inf"] = int(len(X))

    colsC = [f for f in feats if f not in DROP_C]
    h = pd.util.hash_pandas_object(X[colsC].astype(np.float32), index=False).to_numpy()
    counts = pd.DataFrame({"h": h, "y": lab}).value_counts().reset_index(name="n")
    major = counts.sort_values(["h", "n"], ascending=[True, False]).drop_duplicates("h").set_index("h")["y"]
    n_labels = counts.groupby("h")["y"].nunique()
    new = pd.Series(h).map(major).to_numpy()
    changed = new != lab
    stats["rows_relabelled"] = int(changed.sum())
    stats["relabelled_from_to"] = {f"{a} -> {b}": int(n) for (a, b), n in
                                   pd.Series(list(zip(lab[changed], new[changed]))).value_counts().head(20).items()}
    stats["share_rows_in_multilabel_vectors_C"] = round(float((pd.Series(h).map(n_labels).to_numpy() > 1).mean()), 4)

    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(new))).groupby(new).indices
    keep = np.sort(np.concatenate([rng.choice(ix, size=min(len(ix), cap), replace=False) for ix in idx.values()]))
    X, fine = X.iloc[keep].reset_index(drop=True).astype(np.float32), new[keep]
    category = np.array([edgeiiot_category(l) for l in fine])
    stats["cap_per_fine_label"] = cap
    stats["rows_after_sampling"] = int(len(X))
    stats["fine_label_counts"] = {str(k): int(v) for k, v in pd.Series(fine).value_counts().items()}
    stats["unique_vectors_C_in_sample"] = int(pd.util.hash_pandas_object(X[colsC], index=False).nunique())
    print(f"  {stats['rows_after_nan_inf']} satır; yeniden etiketlenen {stats['rows_relabelled']}; örneklem {len(X)} "
          f"(tekil 32-vektör {stats['unique_vectors_C_in_sample']})")
    return X, fine, category, stats
