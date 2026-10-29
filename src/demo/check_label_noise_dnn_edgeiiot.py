"""Demo teşhisi, 2. adım: yerelde normal trafikte alarm alan paket tipleri eğitim verisinde hangi etiketlerle geçiyor?

Gözlem (29.10.2026): Temperature_and_Humidity'nin ilk 20 bin paketinde en sık beş paket tipinin dördü (MQTT sunucusuna boş ACK,
SYN, MQTT CONNECT, MQTT DISCONNECT) WEB senaryosu modelinde %100 ALARM alıyor; tshark sürümü ve scikit-learn elendi (çıkarımlar
birebir aynı). Hipotez: veri setinde etiket dosya bazında verildiği için saldırı kayıtlarına karışan sensör trafiği saldırı
etiketi taşıyor; 34 özellikte seq/ack farklı olduğundan çakışma temizliğinden kaçıyor, koşul C'de birebir aynı hâle geliyor.

Ölçülen (en sık N paket tipi, model girdisi = koşul C'nin 32 özelliği):
  (a) asıl deneyin verisinde (load_data: temizlik + etiket başına 40.000 örneklem, tohum 42): tipin etiket dağılımı;
  (b) aynı temizlikten sonra örneklemsiz tüm veride: etiket dağılımı.
  (c) koşul C'de birden fazla ince etikette görülen vektörlerin satır payı (bütün veri için, etiket başına).

Çalıştırma (Kaggle CPU, Edge-IIoTset ekli):
  python src/demo/check_label_noise_dnn_edgeiiot.py --data-file <DNN-EdgeIIoT-dataset.csv> \\
      --ref outputs/dnn_edgeiiot/demo/kaggle_extract/Normal_Temperature_and_Humidity.csv.gz
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from extractor_dnn_edgeiiot import CONDITIONS, FEATURES


def keys(df, cols):
    return df[cols].astype(np.float64).round(4).astype(str).agg("|".join, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--out", default="outputs/dnn_edgeiiot/demo/label_noise_check.json")
    a = p.parse_args()
    from model_dnn_edgeiiot import load_data
    colsC = [f for f in FEATURES if f not in CONDITIONS["C_seq_ack_yok"]]

    ref = pd.read_csv(a.ref)
    rk = keys(ref, colsC)
    top = rk.value_counts().head(a.top)

    out = {"ref": os.path.basename(a.ref), "top_types": []}
    data = {}
    for name, cap in (("asıl deney verisi (40.000/etiket, tohum 42)", 40000), ("temizlenmiş veri, örneklemsiz", 10**9)):
        X, fine, category, _ = load_data(a.data_file, cap, 42, drop_conflicts=True)
        data[name] = (keys(X, colsC), fine)
    for k, n in top.items():
        row = {"packets_in_ref": int(n), "share_in_ref": round(n / len(ref), 4),
               "values": {c: v for c, v in zip(colsC, k.split("|")) if float(v) != 0}}
        for name, (dk, fine) in data.items():
            hit = (dk == k).to_numpy()
            row[name] = {str(lb): int(c) for lb, c in pd.Series(fine[hit]).value_counts().items()}
        out["top_types"].append(row)
        print(f"\n{n} paket (%{n / len(ref) * 100:.1f}) sıfır olmayan değerler {row['values']}")
        for name in data:
            print(f"   {name}: {row[name] or 'YOK'}")

    # (c) koşul C'de çok etiketli vektörlerin payı
    dk, fine = data["temizlenmiş veri, örneklemsiz"]
    g = pd.DataFrame({"k": dk.to_numpy(), "y": fine})
    multi = g.groupby("k")["y"].nunique()
    is_multi = g["k"].map(multi).to_numpy() > 1
    share = pd.Series(is_multi).groupby(g["y"].to_numpy()).mean().round(4).to_dict()
    out["multi_label_share_by_label_condition_C"] = {str(k): float(v) for k, v in share.items()}
    print("\nKoşul C'de başka bir ince etiketle de görülen vektörlerin satır payı (temizlenmiş veri):")
    for k, v in sorted(share.items(), key=lambda t: -t[1]):
        print(f"   {k:22s} %{v * 100:.1f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {a.out}")


if __name__ == "__main__":
    main()
