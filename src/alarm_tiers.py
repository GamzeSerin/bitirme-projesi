"""KEŞİF (sonuçlardan sonra tanımlandı): alarmı kademelendirmek analist yükünü azaltır mı?

dss_guard.py'de TabM alarmı her zaman ANALYST_REVIEW veriyor ve normal trafiğe yükün çoğu buradan geliyor.
Bu araç, dss_guard'ın kaydettiği satır dosyalarından (rows_<aile>_<koşul>.npz) yeniden eğitim yapmadan
üç kurulumu yan yana hesaplar:
  tek       : alarm (anlaşmazlık > kalibrasyonda bilinenlerin %95'i) -> ANALYST_REVIEW   (dss_guard ile aynı)
  skor      : alarm skoruna göre iki kademe: %95–%99 arası -> en az LOG_MONITOR, %99 üstü -> ANALYST_REVIEW
              (Gamze Serin'in önerisi; eşikler yalnız kalibrasyondaki bilinen satırlardan)
  ciddiyet  : alarm + DSS sınıfı düşük ciddiyet (BENIGN, RECON; ciddiyet <= 1) -> en az LOG_MONITOR,
              alarm + ciddiyet >= 2 -> ANALYST_REVIEW (ikinci varyant; risk: görülmemiş ağır saldırılar sık sık
              RECON sanılıyor)
"En az LOG_MONITOR": alt kademe yalnız yükseltir, hiçbir kararı düşürmez (DSS BLOCK/ALERT/ANALYST_REVIEW
dediyse o kalır; ALLOW ve LOG_MONITOR -> LOG_MONITOR). Enerji kontrolü (q > 0) alarm almamış ve DSS'in
BENIGN dediği satırlara dss_guard'daki gibi uygulanır ve ANALYST_REVIEW verir.

Çalıştırma (önce dss_guard.py):
  python src/alarm_tiers.py --dataset dataset_23
  python src/alarm_tiers.py --dataset dnn_edgeiiot
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import open_set_registry as registry
from common.taxonomy import SEVERITY

ACTIONS = ["ALLOW", "LOG_MONITOR", "ANALYST_REVIEW", "ALERT", "BLOCK"]
VARIANTS = ["tek", "skor", "ciddiyet"]


def at_least_log(actions):
    return np.where(np.isin(actions, ["ALLOW", "LOG_MONITOR"]), "LOG_MONITOR", actions)


def evaluate(z, benign, q, low, high):
    known = [str(c) for c in z["known"]]
    b = known.index(benign)
    unk = z["is_unknown"]
    ben = z["test_category"] == benign
    cal_ben = z["cal_category"] == benign
    d_te, d_cal = z["disagree_test"], z["disagree_cal"]
    tau_lo, tau_hi = np.quantile(d_cal, low), np.quantile(d_cal, high)
    alarm_te, alarm_cal = d_te > tau_lo, d_cal > tau_lo
    act = z["dss_action_test"].astype(str)
    sev = np.array([SEVERITY[known[i]] for i in z["dss_pred_test"]])

    flag = np.zeros_like(unk)
    if q > 0:
        ref = ~alarm_cal & (z["dss_pred_cal"] == b) & cal_ben
        tau_g = np.quantile(z["energy_cal"][ref], 1 - q)
        flag = ~alarm_te & (z["dss_pred_test"] == b) & (z["energy_test"] > tau_g)

    out = {}
    for v in VARIANTS:
        final = act.copy()
        if v == "tek":
            final[alarm_te] = "ANALYST_REVIEW"
        else:
            top = (d_te > tau_hi) if v == "skor" else (sev >= 2)
            low_tier = alarm_te & ~top
            final[low_tier] = at_least_log(act[low_tier])
            final[alarm_te & top] = "ANALYST_REVIEW"
        final[flag] = "ANALYST_REVIEW"
        out[v] = {"unknown": {a: float((final[unk] == a).mean()) for a in ACTIONS},
                  "benign": {a: float((final[ben] == a).mean()) for a in ACTIONS}}
    return out, {"tau_low": float(tau_lo), "tau_high": float(tau_hi)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset_23", choices=list(registry.DATASETS))
    p.add_argument("--rows-dir", default=None, help="Varsayılan outputs/<dataset>/dss_guard")
    p.add_argument("--condition", default=None, help="Varsayılan: ana koşul")
    p.add_argument("--low", type=float, default=0.95, help="Alarm eşiği (dss_guard --target ile aynı)")
    p.add_argument("--high", type=float, default=0.99, help="Skor varyantında üst kademe eşiği")
    p.add_argument("--qs", default="0,0.05", help="Enerji kontrolü bütçeleri (0 = kontrol yok)")
    args = p.parse_args()
    spec = registry.get(args.dataset)
    cond = args.condition or spec["default_condition"]
    rows_dir = args.rows_dir or f"outputs/{args.dataset}/dss_guard"
    files = sorted(glob.glob(os.path.join(rows_dir, f"rows_*_{cond}.npz")))
    if not files:
        raise SystemExit(f"{rows_dir} altında rows_*_{cond}.npz yok: önce dss_guard.py çalıştırın.")
    qs = [float(x) for x in args.qs.split(",")]

    per_family, thresholds = {}, {}
    for f in files:
        fam = os.path.basename(f)[len("rows_"):-len(f"_{cond}.npz")]
        z = np.load(f, allow_pickle=True)
        per_family[fam] = {}
        for q in qs:
            per_family[fam][str(q)], thresholds[fam] = evaluate(z, spec["benign"], q, args.low, args.high)

    mean = {str(q): {v: {grp: {a: round(float(np.mean([per_family[f][str(q)][v][grp][a] for f in per_family])), 5)
                               for a in ACTIONS} for grp in ("unknown", "benign")} for v in VARIANTS} for q in qs}
    out = {"dataset_id": args.dataset, "condition": cond, "status": "KEŞİF: sonuçlardan sonra tanımlandı",
           "low": args.low, "high": args.high, "variants": VARIANTS, "thresholds": thresholds,
           "mean_over_families": mean, "per_family": per_family}
    path = os.path.join(rows_dir, f"alarm_tiers_{cond}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"{spec['name']}, koşul {cond}, {len(per_family)} aile ortalaması (alarm %{100*args.low:g}, "
          f"üst kademe %{100*args.high:g}):")
    print(f"  {'q':>5s} {'kurulum':9s} | görülmemiş: {'ALLOW':>6s} {'LOG':>6s} {'İNCELE':>6s} {'BLK+ALR':>7s}"
          f" | normal: {'İNCELE':>6s} {'LOG':>6s} {'ALLOW':>6s}")
    for q in qs:
        for v in VARIANTS:
            u, n = mean[str(q)][v]["unknown"], mean[str(q)][v]["benign"]
            print(f"  {q:5.2f} {v:9s} |             {u['ALLOW']:6.3f} {u['LOG_MONITOR']:6.3f} "
                  f"{u['ANALYST_REVIEW']:6.3f} {u['BLOCK'] + u['ALERT']:7.3f} |         "
                  f"{n['ANALYST_REVIEW']:6.3f} {n['LOG_MONITOR']:6.3f} {n['ALLOW']:6.3f}")
    print(f"\nÇıktı: {path}")


if __name__ == "__main__":
    main()
