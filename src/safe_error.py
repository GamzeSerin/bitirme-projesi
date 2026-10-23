"""Görülmemiş saldırının DSS'teki akıbeti: inceleme, güvenli hata ya da sessiz kaçış.

two_signal.py'nin kaydettiği satır dosyalarından (rows_<aile>_<koşul>[_seed<N>].npz) yeniden eğitim yapmadan
hesaplanır. Her görülmemiş saldırı satırı tam olarak bir gruba düşer:
  review    TabM alarmı verdi -> ANALYST_REVIEW
  block     alarm yok, ciddiyeti 3 olan bir saldırı sanıldı (DoS, DDoS, Mirai) -> güvenli hata
  alert     alarm yok, ciddiyeti 1-2 olan bir saldırı sanıldı (RECON, SPOOFING, WEB, BRUTEFORCE) -> güvenli hata
  silent    alarm yok, BENIGN sanıldı -> sessiz kaçış (yalnız alarm kuralı)
Güvenlik kontrolü (iki sinyalli kural) yalnız BENIGN denen satırlara dokunduğu için block/alert oranları
bütçeden bağımsızdır; bütçe q'daki sessiz kaçış two_signal JSON'undan okunur.
safe_conf: güvenli hatanın en yüksek olasılığı >= 0,80 olan kısmı. DSS ciddiyet olasılığını kullanır ve bu
değer en yüksek olasılıktan küçük olamaz; yani bu kısım DSS'te kesin BLOCK/ALERT alır (alt sınır). Kalanı
düşük güvende LOG_MONITOR'a düşebilir; hiçbiri ALLOW almaz.

Çalıştırma (önce two_signal.py üç tohumla çalışmış olmalı):
  python src/safe_error.py --dataset dataset_23
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import open_set_registry as registry
from common.taxonomy import SEVERITY

HIGH_CONF = 0.80  # main.py ile aynı


def seed_files(base, condition, seed):
    if seed is None:
        return base, condition
    return os.path.join(base, f"s{seed}"), f"{condition}_seed{seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset_23", choices=list(registry.DATASETS))
    p.add_argument("--condition", default="C_ttl_yok")
    p.add_argument("--seeds", default="42,1,2", help="42 = varsayılan tohum (alt klasörsüz)")
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--guard", default="tabm_energy")
    args = p.parse_args()

    base = os.path.join("outputs", args.dataset, "two_signal")
    seeds = [int(s) for s in args.seeds.split(",")]
    per_family = {}
    for s in seeds:
        d, tag = seed_files(base, args.condition, None if s == 42 else s)
        with open(os.path.join(d, f"two_signal_{tag}.json"), encoding="utf-8") as f:
            ts = json.load(f)
        for fam in ts["families"]:
            z = np.load(os.path.join(d, f"rows_{fam}_{tag}.npz"))
            u = z["is_unknown"]
            alarm, pred = z["alarm"][u], z["tabm_pred"][u]
            conf = 1.0 - z["score_tabm_msp"][u]
            sev = np.array([SEVERITY[c] for c in pred])
            safe = ~alarm & (sev >= 1)
            q_row = [r for r in ts["results"] if r["family"] == fam and r["guard"] == args.guard
                     and abs(r["q"] - args.q) < 1e-9][0]
            per_family.setdefault(fam, []).append({
                "seed": s, "n_unknown": int(u.sum()),
                "review": float(alarm.mean()),
                "block": float((~alarm & (sev >= 3)).mean()),
                "alert": float((~alarm & (sev >= 1) & (sev <= 2)).mean()),
                "silent": float((~alarm & (sev == 0)).mean()),
                "safe_conf": float((safe & (conf >= HIGH_CONF)).mean()),
                f"silent_q{args.q:g}": q_row["silent_benign"],
            })

    keys = [k for k in per_family[next(iter(per_family))][0] if k not in ("seed", "n_unknown")]
    summary = {fam: {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
               for fam, rows in per_family.items()}
    summary["ORTALAMA"] = {k: round(float(np.mean([summary[f][k] for f in per_family])), 4) for k in keys}

    out = {"dataset_id": args.dataset, "condition": args.condition, "seeds": seeds, "guard": args.guard,
           "q": args.q, "classifier": "TabM (two_signal.py satır dosyaları)",
           "groups": "review = alarm; block/alert = alarm yok, ciddiyeti 3 / 1-2 olan saldırı sanıldı "
                     "(güvenli hata); silent = alarm yok, BENIGN sanıldı",
           "mean_over_seeds": summary, "per_seed": per_family}
    path = os.path.join(base, f"safe_error_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'aile':11s}" + "".join(f"{k:>12s}" for k in keys))
    for fam, m in summary.items():
        print(f"{fam:11s}" + "".join(f"{m[k]:12.3f}" for k in keys))
    print(f"\nÇıktı: {path}")


if __name__ == "__main__":
    main()
