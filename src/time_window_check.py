"""Zaman bölmesinde normal trafiğin pencereler arası bileşimi (ekip ortak aracı, --dataset; model yok, CPU).

Neden (02.11.2026, Edge-IIoTset): zaman bölmesinde yeniden kalibrasyon (two_signal --recal-frac 0.25) bütçeyi geri getirmedi:
test penceresinin ilk %25'iyle (W1) yeniden seçilen kontrol eşiği, q = %0,5 ile %5 arasında W2'de yine aynı yükü veriyor (plato,
3 tohum). Bu, W2'deki normal trafiğin bir kısmının W1'de ve kalibrasyonda HİÇ bulunmayan türden olduğunu düşündürüyor. Hipotez:
normal etiketi birden çok cihazın kaydının art arda eklenmesinden oluşuyor; "kayıt sırası" bu yüzden kısmen CİHAZ sırası ve son
pencereler eğitimde görülmemiş cihazları içeriyor (demodaki Modbus bulgusunun aynısı).

Ölçülen (two_signal'ın zaman bölmesiyle birebir aynı pencereler; yalnız zararsız kategori):
  fit (eğitim havuzunun ilk %90'ı), cal (son %10), W1 (test penceresinin ilk %25'i), W2 (kalan %75).
  - yenilik payı: penceredeki satırların ana koşul vektörü (koşul sütunları çıkarılmış) daha önceki pencerelerin hiçbirinde yoksa
    "yeni" sayılır: cal | fit, W1 | fit+cal, W2 | fit+cal+W1, W2 | fit+cal.
  - parmak izi sütunları: pencereler arasında dağılımı en çok değişen sütunlar (toplam değişim uzaklığı, cal ile W2 arası) ve her
    pencerede en sık değerleri (Edge-IIoTset'te mqtt.topic_len sensör konusunun adına göre değişir: cihaz parmak izi).

Çalıştırma:
  python src/time_window_check.py --dataset dnn_edgeiiot --data-file <DNN-EdgeIIoT-dataset.csv>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import open_set_registry as registry
from two_signal import temporal_split


def novelty(h, mask, seen_mask):
    seen = set(h[seen_mask])
    return round(float(np.mean([x not in seen for x in h[mask]])), 4) if mask.any() else None


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--condition", default=None)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--recal-frac", type=float, default=0.25)
    p.add_argument("--columns", default="", help="Parmak izi sütunları (virgülle); verilmezse en çok değişen 5 sütun")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    spec = registry.get(a.dataset)
    cond = a.condition or spec["default_condition"]
    out_dir = a.out_dir or f"outputs/{a.dataset}/two_signal"
    os.makedirs(out_dir, exist_ok=True)

    X, fine, category, _ = spec["loader"](a.data_file, a.cap, a.seed)
    X, _ = registry.apply_condition(X, spec, cond)
    allidx = np.arange(len(category))
    tr, te = temporal_split(allidx, fine, 0.2)
    fit, cal = temporal_split(tr, fine, a.cal_size)
    w1, w2 = temporal_split(te, fine, 1 - a.recal_frac)
    win = np.full(len(category), "", dtype=object)
    for name, ix in (("fit", fit), ("cal", cal), ("W1", w1), ("W2", w2)):
        win[ix] = name
    ben = category == spec["benign"]
    h = pd.util.hash_pandas_object(X.astype(np.float32), index=False).to_numpy()
    m = {w: ben & (win == w) for w in ("fit", "cal", "W1", "W2")}
    out = {"dataset_id": a.dataset, "condition": cond, "benign_fine_labels": sorted(set(fine[ben])),
           "rows": {w: int(v.sum()) for w, v in m.items()},
           "novelty": {"cal|fit": novelty(h, m["cal"], m["fit"]),
                       "W1|fit+cal": novelty(h, m["W1"], m["fit"] | m["cal"]),
                       "W2|fit+cal+W1": novelty(h, m["W2"], m["fit"] | m["cal"] | m["W1"]),
                       "W2|fit+cal": novelty(h, m["W2"], m["fit"] | m["cal"])}}
    print(f"{spec['name']} · koşul {cond} · zararsız satırlar {out['rows']}")
    print("Yeni vektör payı (önceki pencerelerde hiç görülmemiş):")
    for k, v in out["novelty"].items():
        print(f"   {k:16s} %{v * 100:.1f}" if v is not None else f"   {k:16s} -")

    Xb = X[ben]
    wb = win[ben]
    if a.columns:
        cols = [c for c in a.columns.split(",") if c]
    else:
        d = {c: tvd(Xb[c][wb == "cal"], Xb[c][wb == "W2"]) for c in X.columns}
        cols = [c for c, _ in sorted(d.items(), key=lambda t: -t[1])[:5]]
    out["fingerprint"] = {}
    for c in cols:
        r = {"tvd_cal_W2": round(tvd(Xb[c][wb == "cal"], Xb[c][wb == "W2"]), 4),
             "tvd_W1_W2": round(tvd(Xb[c][wb == "W1"], Xb[c][wb == "W2"]), 4)}
        print(f"\n{c}  (TVD cal–W2 {r['tvd_cal_W2']:.3f}, W1–W2 {r['tvd_W1_W2']:.3f})")
        for w in ("fit", "cal", "W1", "W2"):
            vc = Xb[c][wb == w].value_counts(normalize=True).head(a.top)
            r[w] = {str(float(k)): round(float(v), 4) for k, v in vc.items()}
            print(f"   {w:4s} " + "  ".join(f"{float(k):g}: %{v * 100:.1f}" for k, v in vc.items()))
        out["fingerprint"][c] = r
    path = os.path.join(out_dir, f"time_window_check_{cond}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {path}")


if __name__ == "__main__":
    main()
