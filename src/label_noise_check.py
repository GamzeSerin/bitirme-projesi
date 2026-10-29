"""Etiket karışıklığı payı (ekip ortak aracı, --dataset): ana koşulun özellikleriyle birebir aynı vektörün başka bir ince etiketle de
görüldüğü satırların payı.

Edge-IIoTset'te bulundu (29.10.2026, demo teşhisi): koşul C'de (seq/ack yok) Backdoor %89, Ransomware %85, XSS %79, Password %60,
Normal %11 (`outputs/dnn_edgeiiot/demo/label_noise_check.json`). Paket/akış düzeyinde etiketler kayıt dosyası bazında verildiği
için saldırı kayıtlarındaki sıradan trafik saldırı etiketi taşıyabiliyor; asıl deneyin çakışma temizliği bunu tüm özelliklerle
yaptığı için koşulda çıkarılan sütunlar gidince çakışmalar yeniden ortaya çıkabiliyor. Bu araç aynı ölçümü her veri setinde yapar.

Ölçülen, iki veri üzerinde: (a) asıl deneyin verisi (kayıt yükleyicisi, etiket başına --cap örneklem, tohum 42) ve
(b) aynı temizlikten sonra örneklemsiz veri (--cap-full). Koşulun sütunları çıkarıldıktan sonra her satırın vektörü hash'lenir;
vektör birden fazla ince etikette görülüyorsa satır "karışık" sayılır. Rapor: ince etiket ve ortak kategori başına karışık satır
payı ve en sık karıştığı etiketler.

Çalıştırma:
  python src/label_noise_check.py --dataset dataset_23 --data-file data/dataset_23
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import open_set_registry as registry


def mixed_share(X, fine, category):
    h = pd.util.hash_pandas_object(X.astype(np.float32), index=False).to_numpy()
    g = pd.DataFrame({"h": h, "y": fine, "c": category})
    n_lab = g.groupby("h")["y"].nunique()
    mixed = g["h"].map(n_lab).to_numpy() > 1
    by_fine = pd.Series(mixed).groupby(fine).mean().round(4)
    by_cat = pd.Series(mixed).groupby(category).mean().round(4)
    partners = {}
    for lab in np.unique(fine):
        hs = set(g.loc[(g["y"] == lab).to_numpy() & mixed, "h"])
        other = g[g["h"].isin(hs) & (g["y"] != lab)]["y"].value_counts(normalize=True).head(3)
        partners[str(lab)] = {str(k): round(float(v), 3) for k, v in other.items()}
    return {"rows": int(len(g)), "unique_vectors": int(g["h"].nunique()), "mixed_share_all": round(float(mixed.mean()), 4),
            "by_fine_label": {str(k): float(v) for k, v in by_fine.items()},
            "by_category": {str(k): float(v) for k, v in by_cat.items()}, "mixed_with": partners}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--condition", default=None)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--cap-full", type=int, default=10**9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/label_noise")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.cap, a.cap_full = 2000, 4000
    spec = registry.get(a.dataset)
    cond = a.condition or spec["default_condition"]
    out_dir = a.out_dir or f"outputs/{a.dataset}/label_noise"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    out = {"dataset_id": a.dataset, "condition": cond, "dropped_by_condition": spec["conditions"][cond]}
    for name, cap in (("experiment_sample", a.cap), ("cleaned_full", a.cap_full)):
        X, fine, category, _ = spec["loader"](a.data_file, cap, a.seed)
        X, _ = registry.apply_condition(X, spec, cond)
        out[name] = r = mixed_share(X, fine, category)
        print(f"\n{spec['name']} · {name} (etiket başına en fazla {cap}): {r['rows']} satır, "
              f"karışık pay %{r['mixed_share_all'] * 100:.1f}")
        for k, v in sorted(r["by_fine_label"].items(), key=lambda t: -t[1]):
            print(f"   {k:28s} %{v * 100:5.1f}   en sık karıştığı: {r['mixed_with'][k]}")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(out_dir, f"label_noise_{cond}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {path}")


if __name__ == "__main__":
    main()
