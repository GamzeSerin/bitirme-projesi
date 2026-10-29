"""Derin öğrenme karşılaştırması: TabM / MLP / MC-dropout (iki sinyal kuralı, ekip ortak aracı, --dataset).

two_signal.py çıktılarını okur (her sınıflandırıcı ve model tohumu için), tabloyu ve figürü üretir. Analiz planı
sonuçlardan önce yazıldı (data/dnn_edgeiiot/README.md, "Derin öğrenme karşılaştırması"); eşik yok.

TabM'in mevcut koşularında AUROC ve macro-F1 JSON'da yoksa kayıtlı satır dosyalarından aynı fonksiyonla
(two_signal.extra_metrics) hesaplanır.

Çalıştırma:
  python src/dl_compare.py --dataset dnn_edgeiiot
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from two_signal import CLASSIFIERS, extra_metrics

METRICS = ["alarm_auroc", "energy_auroc", "known_macro_f1"]
QS = [0.005, 0.01, 0.02, 0.05, 0.10]


def run_path(base, cond, clf, seed, split_seed=42):
    tag = cond if clf == "TabM" else f"{cond}_{clf.lower()}"
    d = base if seed == split_seed else os.path.join(base, f"s{seed}")
    for t in ([tag] if seed == split_seed else [f"{tag}_seed{seed}", tag]):   # eski TabM tohum dosyaları etiketsiz
        p = os.path.join(d, f"two_signal_{t}.json")
        if os.path.exists(p):
            return p, d, t
    return None, d, None


def tabm_metrics_from_rows(d, tag, fam, spec):
    p = os.path.join(d, f"rows_{fam}_{tag}.npz")
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    cats = set(z["category"])
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    known = [c for c in present if c not in spec["families"][fam]]
    return extra_metrics(z["is_unknown"], z["score_tabm_disagree"], z["score_tabm_energy"],
                         z["tabm_pred"], z["category"], known)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--condition", default=None)
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--base-dir", default=None, help="Varsayılan outputs/<dataset>/two_signal")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/dl_compare")
    args = p.parse_args()
    spec = registry.get(args.dataset)
    cond = args.condition or spec["default_condition"]
    base = args.base_dir or f"outputs/{args.dataset}/two_signal"
    out_dir = args.out_dir or f"outputs/{args.dataset}/dl_compare"
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    per = {}   # clf -> seed -> {"mean": {...}, "families": {fam: {...}}, "train_seconds": ...}
    for clf in CLASSIFIERS:
        prefix = CLASSIFIERS[clf][0]
        for seed in seeds:
            path, d, tag = run_path(base, cond, clf, seed)
            if path is None:
                print(f"  yok: {clf} tohum {seed}")
                continue
            r = json.load(open(path, encoding="utf-8"))
            fams = {}
            for fam, fd in r["families"].items():
                m = {k: fd[k] for k in METRICS if k in fd}
                if not m and clf == "TabM":
                    m = tabm_metrics_from_rows(d, tag, fam, spec)
                m["silent_alarm_only"] = fd["alarm_only"]["silent_benign"]
                m["train_seconds"] = fd.get("train_seconds")
                for row in r["results"]:
                    if row["family"] == fam and row["guard"] == f"{prefix}_energy":
                        m[f"silent_q{row['q']}"] = row["silent_benign"]
                        m[f"review_q{row['q']}"] = row["benign_review"]
                        m[f"caught_q{row['q']}"] = row["unknown_caught"]
                fams[fam] = m
            keys = sorted({k for m in fams.values() for k in m if fams[next(iter(fams))].get(k) is not None})
            mean = {k: float(np.mean([m[k] for m in fams.values() if m.get(k) is not None])) for k in keys}
            per.setdefault(clf, {})[seed] = {"file": path, "families": fams, "mean": mean}

    # özet: tohumlar arası ortalama ± sd; TabM'e göre tohum başına fark
    summary = {}
    for clf, by_seed in per.items():
        keys = sorted(set.intersection(*[set(v["mean"]) for v in by_seed.values()]))
        summary[clf] = {"seeds": sorted(by_seed),
                        **{k: {"mean": round(float(np.mean([v["mean"][k] for v in by_seed.values()])), 5),
                               "sd": round(float(np.std([v["mean"][k] for v in by_seed.values()], ddof=1)), 5)
                               if len(by_seed) > 1 else None} for k in keys}}
        if clf != "TabM" and "TabM" in per:
            for k in ["alarm_auroc", "silent_q0.05", "known_macro_f1"]:
                diffs = [per[clf][s]["mean"][k] - per["TabM"][s]["mean"][k] for s in by_seed
                         if s in per["TabM"] and k in per[clf][s]["mean"] and k in per["TabM"][s]["mean"]]
                if diffs:
                    summary[clf][f"diff_vs_TabM_{k}"] = {"per_seed": [round(x, 5) for x in diffs],
                                                         "mean": round(float(np.mean(diffs)), 5)}
    fam_table = {clf: {fam: {k: round(float(np.mean([by_seed[s]["families"][fam][k] for s in by_seed
                                                       if by_seed[s]["families"][fam].get(k) is not None])), 5)
                             for k in by_seed[next(iter(by_seed))]["families"][fam]
                             if by_seed[next(iter(by_seed))]["families"][fam].get(k) is not None}
                       for fam in by_seed[next(iter(by_seed))]["families"]}
                 for clf, by_seed in per.items()}
    out = {"dataset_id": args.dataset, "condition": cond, "primary_guard": "enerji (her sınıflandırıcının kendi)",
           "status": "Analiz planı sonuçlardan önce yazıldı (README, Derin öğrenme karşılaştırması); eşik yok",
           "summary": summary, "per_family_mean_over_seeds": fam_table,
           "runs": {clf: {s: v["file"] for s, v in by.items()} for clf, by in per.items()}}
    path = os.path.join(out_dir, f"dl_compare_{cond}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    for clf, sm in summary.items():
        print(f"{clf:10s} tohum {sm['seeds']}: alarm AUROC {sm['alarm_auroc']['mean']:.4f}±{sm['alarm_auroc']['sd'] or 0:.4f}"
              f"  enerji AUROC {sm['energy_auroc']['mean']:.4f}  macro-F1 {sm['known_macro_f1']['mean']:.4f}"
              f"  kontrolsüz kaçış {sm['silent_alarm_only']['mean']:.4f}  q=%5 kaçış {sm['silent_q0.05']['mean']:.4f}"
              f" / inceleme {sm['review_q0.05']['mean']:.4f}")
    plot(summary, os.path.join(out_dir, f"fig_dl_compare_{args.dataset}_{cond}.png"), spec["name"], cond)
    print(f"Çıktı: {path}")


def plot(summary, path, name, cond):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"TabM": "#2a78d6", "MLP": "#9aa5b1", "MCDropout": "#d6762a"}
    labels = {"TabM": "TabM (k = 32)", "MLP": "MLP (k = 1, alarm MSP)", "MCDropout": "MC-dropout (T = 32)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for clf, sm in summary.items():
        rv = [sm[f"review_q{q}"]["mean"] * 100 for q in QS if f"review_q{q}" in sm]
        sl = [sm[f"silent_q{q}"]["mean"] * 100 for q in QS if f"silent_q{q}" in sm]
        a1.plot(rv, sl, marker="o", color=colors[clf], label=labels[clf])
        a1.axhline(sm["silent_alarm_only"]["mean"] * 100, color=colors[clf], ls=":", lw=0.9)
    a1.set_xlabel("Analiste giden normal trafik (%)")
    a1.set_ylabel("Sessiz kaçış (%)")
    a1.set_title("Enerji kontrolü, q = %0,5–10 (noktalı: kontrolsüz)", fontsize=9)
    a1.legend(frameon=False, fontsize=8)
    clfs = list(summary)
    x = np.arange(3)
    w = 0.8 / len(clfs)
    for i, clf in enumerate(clfs):
        sm = summary[clf]
        vals = [sm[k]["mean"] for k in METRICS]
        errs = [sm[k]["sd"] or 0 for k in METRICS]
        a2.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, color=colors[clf], label=labels[clf], capsize=2)
    a2.set_xticks(x)
    a2.set_xticklabels(["alarm AUROC", "enerji AUROC", "bilinen macro-F1"], fontsize=8)
    a2.set_ylim(0.5, 1.0)
    a2.set_title("Ayırma ve sınıflandırma (tohum ort. ± sd)", fontsize=9)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.suptitle(f"{name}, koşul {cond}: TabM topluluğu gerekli mi?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
