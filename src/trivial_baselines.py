"""Baz modeller (#18): tezin sayılarını hiçbir şey öğrenmeyen kurallarla aynı test satırlarında kıyaslar.

Analiz planı (sonuçlardan önce, 26.11.2026): data/dnn_edgeiiot/README.md, "Baz modeller: çoğunluk sınıfı,
'her şeye normal' ve rastgele inceleme".

Model eğitimi yok, veri seti gerekmez. Girdi: two_signal.py'nin kaydettiği satır dosyaları
<runs>/rows_<AİLE>_<koşul>.npz (category, is_unknown, tabm_pred, alarm) ve aynı klasördeki
two_signal_<koşul>.json (enerji kontrolünün gerçekleşen yükü ve sessiz kaçışı).

Kurallar (plandaki gibi, sabit):
  1. Çoğunluk sınıfı: her ailede en sık bilinen kategoriyi her bilinen test satırına söyler; macro-F1 bilinen test
     satırlarında, two_signal.py ile aynı tanım (labels = bilinen kategoriler). Eğitim dağılımı satır dosyalarında
     yok; bölme katmanlı olduğundan en sık kategori test satırlarından alınır.
  2. "Her şeye normal": hiç alarm ve kontrol yok; sessiz kaçış %100, analist yükü %0 (tanım gereği).
  3. Rastgele inceleme: alarm aynı (TabM anlaşmazlığı); alarm almayıp BENIGN denen trafiğin rastgele r kadarı
     analiste gider. r = enerji kontrolünün gerçekleşen yükü (benign_review) / normal trafiğin alarm almayıp BENIGN
     denen payı; yani iki kural normal trafikte aynı yükü taşır. Beklenen sessiz kaçış = kontrolsüz kaçış x (1 - r)
     (kapalı formül); ayrıca 1.000 rastgele çekilişle %95 aralığı (tohum 0).
Ana ölçüt (plandan): q = %5'te enerji / rastgele sessiz kaçış oranı, 5 aile ortalamasında, 3 tohumun HER BİRİNDE
en az 2 kat (>= 2: sağlandı; 1-2: kısmen; < 1: katkı yok).

Çıktı: <out_dir>/trivial_baselines_<koşul>.json (+ "runtime" alanı).
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.contract import environment_info  # noqa: E402

FAMILIES = ["DDOS", "RECON", "WEB", "BRUTEFORCE", "MALWARE"]
BENIGN = "BENIGN"
GUARD = "tabm_energy"


def one_seed(run_dir, condition, n_draws, rng):
    with open(os.path.join(run_dir, f"two_signal_{condition}.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    energy = {(r["family"], r["q"]): r for r in res["results"] if r["guard"] == GUARD}
    qs = sorted({q for (_, q) in energy})
    fams, rows = {}, []
    for fam in FAMILIES:
        z = np.load(os.path.join(run_dir, f"rows_{fam}_{condition}.npz"), allow_pickle=True)
        cat, unk, pred, alarm = z["category"], z["is_unknown"], z["tabm_pred"], z["alarm"]
        kn = ~unk
        known = sorted(set(cat[kn]))
        cats, counts = np.unique(cat[kn], return_counts=True)
        major = str(cats[np.argmax(counts)])
        to_guard = ~alarm & (pred == BENIGN)
        ben = kn & (cat == BENIGN)
        silent0 = float((to_guard & unk).sum() / unk.sum())          # kontrolsüz sessiz kaçış
        ben_reach = float((to_guard & ben).sum() / ben.sum())         # normalin kontrole ulaşan payı
        fams[fam] = {
            "n_known_test": int(kn.sum()), "n_unknown_test": int(unk.sum()), "n_benign_test": int(ben.sum()),
            "known_categories": known, "majority_category": major,
            "majority_macro_f1": round(float(f1_score(cat[kn], np.full(kn.sum(), major), labels=known,
                                                      average="macro", zero_division=0)), 4),
            "tabm_macro_f1": round(float(f1_score(cat[kn], pred[kn], labels=known, average="macro",
                                                  zero_division=0)), 4),
            "silent_alarm_only": round(silent0, 5),
            "silent_alarm_only_json": res["families"][fam]["alarm_only"]["silent_benign"],
            "benign_reach_guard": round(ben_reach, 5)}
        n_sil = int((to_guard & unk).sum())
        for q in qs:
            e = energy[(fam, q)]
            r = min(e["benign_review"] / ben_reach, 1.0) if ben_reach > 0 else 0.0
            draws = rng.binomial(n_sil, 1 - r, size=n_draws) / unk.sum()  # her kaçan satır bağımsız 1-r olasılıkla geçer
            rows.append({"family": fam, "q": q, "load": e["benign_review"], "r": round(r, 5),
                         "silent_energy": e["silent_benign"], "silent_random": round(silent0 * (1 - r), 5),
                         "silent_random_ci95": [round(float(np.percentile(draws, 2.5)), 5),
                                                round(float(np.percentile(draws, 97.5)), 5)]})
    means = {}
    for q in qs:
        sub = [x for x in rows if x["q"] == q]
        se = float(np.mean([x["silent_energy"] for x in sub]))
        sr = float(np.mean([x["silent_random"] for x in sub]))
        means[str(q)] = {"load": round(float(np.mean([x["load"] for x in sub])), 5),
                         "silent_energy": round(se, 5), "silent_random": round(sr, 5),
                         "ratio_random_over_energy": round(sr / se, 2) if se > 0 else None}
    return {"families": fams, "results": rows, "mean_over_families": means,
            "majority_macro_f1_mean": round(float(np.mean([f["majority_macro_f1"] for f in fams.values()])), 4),
            "tabm_macro_f1_mean": round(float(np.mean([f["tabm_macro_f1"] for f in fams.values()])), 4)}


def verdict(ratios):
    if all(r is not None and r >= 2 for r in ratios):
        return "sağlandı (her tohumda >= 2 kat)"
    if all(r is not None and r >= 1 for r in ratios):
        return "kısmen (1-2 kat; kazancın bir kısmı yalnız bütçeden)"
    return "katkı yok (en az bir tohumda < 1 kat)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", default=["outputs/dnn_edgeiiot/two_signal",
                                                  "outputs/dnn_edgeiiot/two_signal/s1",
                                                  "outputs/dnn_edgeiiot/two_signal/s2"])
    ap.add_argument("--condition", default="C_seq_ack_yok")
    ap.add_argument("--out-dir", default="outputs/dnn_edgeiiot/trivial_baselines")
    ap.add_argument("--n-draws", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = {}
    for d in args.runs:
        with open(os.path.join(d, f"two_signal_{args.condition}.json"), encoding="utf-8") as fh:
            seed = json.load(fh)["settings"].get("model_seed") or 42  # tohum 42 koşusunda alan boş
        seeds[str(seed)] = one_seed(d, args.condition, args.n_draws, rng)
        s = seeds[str(seed)]
        print(f"\n=== model tohumu {seed} ({d})")
        print(f"macro-F1 (5 aile ort.): çoğunluk {s['majority_macro_f1_mean']}  TabM {s['tabm_macro_f1_mean']}")
        for fam, f in s["families"].items():
            print(f"  {fam:10s} çoğunluk={f['majority_category']:9s} F1 {f['majority_macro_f1']:.4f} / TabM "
                  f"{f['tabm_macro_f1']:.4f}  kontrolsüz kaçış {f['silent_alarm_only']:.4f} "
                  f"(JSON {f['silent_alarm_only_json']})  normalin kontrole ulaşan payı {f['benign_reach_guard']:.4f}")
        for q, m in s["mean_over_families"].items():
            print(f"  q={q:5s} yük {m['load']:.4f}  kaçış enerji {m['silent_energy']:.5f}  rastgele "
                  f"{m['silent_random']:.5f}  oran {m['ratio_random_over_energy']}")
    ratios = [s["mean_over_families"]["0.05"]["ratio_random_over_energy"] for s in seeds.values()]
    out = {"dataset_id": "dnn_edgeiiot", "condition": args.condition, "guard": GUARD,
           "plan": "data/dnn_edgeiiot/README.md, 'Baz modeller' (26.11.2026, sonuçlardan önce)",
           "always_benign": {"silent_benign": 1.0, "benign_load": 0.0, "note": "tanım gereği"},
           "seeds": seeds, "criterion_q": 0.05, "criterion_ratios": ratios, "verdict": verdict(ratios),
           "settings": {k: v for k, v in vars(args).items()}}
    out["runtime"] = environment_info()  # yazılım sürümleri ve donanım (common/contract.py)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"trivial_baselines_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nAna ölçüt (q=%5, rastgele/enerji kaçış oranı, tohum başına): {ratios} -> {verdict(ratios)}\n{path}")


if __name__ == "__main__":
    main()
