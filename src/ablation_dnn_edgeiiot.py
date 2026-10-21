"""Kişi 3 - Edge-IIoTset kısayol (shortcut) ablation deneyi.

Soru: Modelin ~0,995 macro-F1'i saldırı davranışından mı, yoksa veri toplama
izlerinden (hedef port, TCP sıra/onay numarası, sayaç ve zaman damgası) mı geliyor?

Yöntem: Satırlar ve 80/20 bölme (seed=42) bütün senaryolarda AYNIDIR; yalnızca
sütunlar çıkarılır. Hiperparametreler sabittir (asıl deneyde RF için seçilenler),
böylece fark yalnızca özelliklerden gelir. Her senaryo için:
  - test macro-F1 ve %95 bootstrap güven aralığı, sınıf başına F1, ince etiket recall'ı
  - en önemli 10 özellik (RF impurity önemi)
  - indirgenmiş özellik uzayında birbirinin aynısı hâline gelen ve etiketi çakışan
    satır oranı (özellikler çıkarılınca hangi saldırılar normal trafikten ayırt
    edilemez hâle geliyor?)
Ayrıca şüpheli iki sütun için tek-özellik sızıntı testi yapılır.

Çıktı: outputs/dnn_edgeiiot/ablation/ (ablation.json + 2 grafik). Sözleşme dosyalarına
dokunmaz; DSS etkilenmez.

Kullanım (Kaggle, CPU yeterli, ~5-10 dk):
  python src/ablation_dnn_edgeiiot.py --data-file "<.../DNN-EdgeIIoT-dataset.csv>"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier

from common.taxonomy import COMMON_CLASSES
from model_dnn_edgeiiot import load_data

OUT_DIR = "outputs/dnn_edgeiiot/ablation"

PORTS = ["tcp.dstport", "udp.port"]
SEQ_ACK = ["tcp.seq", "tcp.ack"]
# 19.10.2026 incelemesi: Modbus işlem numarası sürekli artan bir sayaç (udp.stream gibi),
# ICMP zaman damgası kaydın alındığı anı taşıyabilir -> ikisi de veri toplama izi adayı.
SUSPECT_IDS = ["mbtcp.trans_id", "icmp.transmit_timestamp"]

SCENARIOS = {
    "A_tam": [],
    "B_portsuz": PORTS,
    "C_seq_ack_yok": SEQ_ACK,
    "D_sayac_zaman_yok": SUSPECT_IDS,
    "E_hepsi_cikarildi": PORTS + SEQ_ACK + SUSPECT_IDS,
}
SCENARIO_TITLES = {
    "A_tam": "A: Tam (34 özellik)",
    "B_portsuz": "B: Portlar yok",
    "C_seq_ack_yok": "C: tcp.seq/ack yok",
    "D_sayac_zaman_yok": "D: Sayaç + zaman damgası yok",
    "E_hepsi_cikarildi": "E: Hepsi çıkarıldı",
}


def build_models(seed, n_jobs):
    # Asıl deneyde (18.10.2026, --tune) RF ve DT için seçilen parametreler sabitlendi.
    return {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=40, max_features=0.5,
                                     class_weight="balanced_subsample",
                                     n_jobs=n_jobs, random_state=seed),
        "DT": DecisionTreeClassifier(max_depth=20, criterion="entropy",
                                     class_weight="balanced", random_state=seed),
    }


def bootstrap_macro_f1(y_true, y_pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [f1_score(y_true[ix], y_pred[ix], average="macro")
              for ix in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def collision_stats(X, fine, category):
    """İndirgenmiş uzayda: kaç satırın aynısı başka bir satırda var, kaçı FARKLI kategoriyle."""
    df = X.copy()
    df["_cat"] = category
    feats = list(X.columns)
    dup = df.duplicated(subset=feats, keep=False).to_numpy()
    n_cat = df.groupby(feats, sort=False)["_cat"].transform("nunique").to_numpy()
    conflict = n_cat > 1
    by_label = pd.Series(fine[conflict]).value_counts()
    totals = pd.Series(fine).value_counts()
    return {
        "duplicate_rows_pct": round(100 * float(dup.mean()), 2),
        "cross_category_conflict_rows": int(conflict.sum()),
        "cross_category_conflict_pct": round(100 * float(conflict.mean()), 2),
        "conflict_pct_by_fine_label": {k: round(100 * int(v) / int(totals[k]), 2)
                                       for k, v in by_label.items()},
    }


def single_feature_leak(X, y, cols, seed, per_class=5000):
    """Aşama 0'daki test: tek sütunla derinlik-8 ağaç, 3-fold macro-F1."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], size=min(per_class, (y == c).sum()),
                                     replace=False) for c in np.unique(y)])
    out = {}
    for c in cols:
        if c not in X.columns:
            continue
        col = X[c].iloc[idx]
        pred = cross_val_predict(DecisionTreeClassifier(max_depth=8, random_state=seed),
                                 col.to_frame(), y[idx], cv=3)
        out[c] = {
            "macro_f1_alone": round(float(f1_score(y[idx], pred, average="macro")), 4),
            "n_unique": int(X[c].nunique()),
            "nonzero_pct_by_category": {},
        }
    return out


def plot(results, classes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    models = list(results[names[0]]["models"])
    colors = {"RF": "#2a78d6", "DT": "#9aa5b1"}

    fig, ax = plt.subplots(figsize=(8, 4.2))
    h = 0.8 / len(models)
    for j, m in enumerate(models):
        ys = np.arange(len(names)) + (j - (len(models) - 1) / 2) * h
        vals = [results[s]["models"][m]["test_macro_f1"] for s in names]
        lo = [v - results[s]["models"][m]["test_macro_f1_ci95"][0] for v, s in zip(vals, names)]
        hi = [results[s]["models"][m]["test_macro_f1_ci95"][1] - v for v, s in zip(vals, names)]
        ax.barh(ys, vals, height=h * 0.9, color=colors.get(m, "#555"), label=m,
                xerr=[lo, hi], error_kw={"elinewidth": 0.8, "capsize": 2})
        for yv, v, e in zip(ys, vals, hi):
            ax.text(v + e + 0.006, yv, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(names)), [SCENARIO_TITLES[s] for s in names])
    ax.invert_yaxis()
    lo_all = min(results[s]["models"][m]["test_macro_f1"] for s in names for m in models)
    ax.set_xlim(max(0, lo_all - 0.08), 1.06)
    ax.set_xlabel("Test macro-F1 (%95 bootstrap GA)")
    ax.set_title("Edge-IIoTset: özellik çıkarma (ablation)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_ablation_macro_f1_dnn_edgeiiot_{'_'.join(models)}.png"), dpi=200)
    plt.close(fig)

    mat = np.array([[results[s]["models"]["RF"]["test_per_class_f1"][c] for c in classes]
                    for s in names])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(mat, cmap="Blues", vmin=max(0, mat.min() - 0.05), vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [SCENARIO_TITLES[s] for s in names])
    for i in range(len(names)):
        for j in range(len(classes)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > (mat.min() + 1) / 2 + 0.02 else "#14202b")
    ax.set_title("RF: senaryo × sınıf F1", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_ablation_class_f1_dnn_edgeiiot_RF.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--models", default="RF,DT")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Etiket başına 3.000 satır, deneme için")
    args = p.parse_args()
    if args.quick:
        args.cap, args.n_boot = min(args.cap, 3000), 50
    os.makedirs(args.out_dir, exist_ok=True)

    # Asıl deneyle aynı satırlar: tam özellik uzayında tekrar ve çakışma temizliği.
    X, fine, category, data_stats = load_data(args.data_file, args.cap, args.seed,
                                              drop_features=(), drop_conflicts=True)
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=fine, random_state=args.seed)
    print(f"Örneklem: {len(y)} satır, {X.shape[1]} özellik; test {len(idx_te)} satır")

    leak = single_feature_leak(X, y, SUSPECT_IDS, args.seed)
    for c in leak:
        nz = (X[c] != 0).groupby(category).mean()
        leak[c]["nonzero_pct_by_category"] = {k: round(100 * float(v), 2) for k, v in nz.items()}
        print(f"Tek-özellik testi {c}: macro-F1={leak[c]['macro_f1_alone']}, "
              f"farklı değer={leak[c]['n_unique']}")

    models = build_models(args.seed, args.n_jobs)
    model_names = [m for m in args.models.split(",") if m in models]
    results = {}
    for s in args.scenarios.split(","):
        drop = [c for c in SCENARIOS[s] if c in X.columns]
        Xs = X.drop(columns=drop)
        print(f"\n== {SCENARIO_TITLES[s]} ({Xs.shape[1]} özellik) ==")
        res = {"dropped": drop, "n_features": int(Xs.shape[1]),
               "collisions": collision_stats(Xs, fine, category), "models": {}}
        print(f"  farklı kategoriyle çakışan satır: %{res['collisions']['cross_category_conflict_pct']}")
        for m in model_names:
            t0 = time.perf_counter()
            est = clone(models[m]).fit(Xs.iloc[idx_tr], y[idx_tr])
            fit_s = time.perf_counter() - t0
            pred = est.predict(Xs.iloc[idx_te])
            yte, fte = y[idx_te], fine[idx_te]
            per_class = f1_score(yte, pred, average=None, labels=range(len(classes)))
            imp = pd.Series(est.feature_importances_, index=Xs.columns).nlargest(10)
            res["models"][m] = {
                "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
                "test_macro_f1_ci95": bootstrap_macro_f1(yte, pred, args.n_boot, args.seed),
                "test_per_class_f1": {c: round(float(v), 4) for c, v in zip(classes, per_class)},
                "test_fine_label_recall": {lab: round(float((pred[fte == lab] == yte[fte == lab]).mean()), 4)
                                           for lab in sorted(set(fte))},
                "top10_importance": {k: round(float(v), 4) for k, v in imp.items()},
                "fit_seconds": round(fit_s, 2),
            }
            r = res["models"][m]
            print(f"  {m}: macro-F1={r['test_macro_f1']} {r['test_macro_f1_ci95']}  "
                  f"en önemli: {list(r['top10_importance'])[:3]}")
        results[s] = res

    base = results.get("A_tam")
    if base:
        for s, r in results.items():
            for m, mr in r["models"].items():
                mr["delta_vs_full"] = round(mr["test_macro_f1"] - base["models"][m]["test_macro_f1"], 4)

    out = {
        "dataset_id": "dnn_edgeiiot",
        "dataset_name": "Edge-IIoTset",
        "protocol": "Aynı satırlar ve aynı 80/20 bölme (seed=42); sabit hiperparametreler; "
                    "yalnızca sütunlar çıkarılır. Satırlar tam özellik uzayında tekrar/çakışma "
                    "temizliğinden geçti (--drop-conflicts).",
        "seed": args.seed, "cap_per_fine_label": args.cap, "quick": bool(args.quick),
        "n_rows": int(len(y)), "n_test": int(len(idx_te)), "classes": classes,
        "scenarios": SCENARIOS,
        "single_feature_leak_test": leak,
        "results": results,
        "data_prep": data_stats,
    }
    with open(os.path.join(args.out_dir, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if "RF" in model_names:
        plot(results, classes, args.out_dir)

    print("\nÖzet (test macro-F1, tam modele göre fark):")
    for s, r in results.items():
        row = "  ".join(f"{m} {mr['test_macro_f1']:.4f} ({mr.get('delta_vs_full', 0):+.4f})"
                        for m, mr in r["models"].items())
        print(f"  {SCENARIO_TITLES[s]:32s} {row}")
    print(f"Çıktılar: {args.out_dir}")


if __name__ == "__main__":
    main()
