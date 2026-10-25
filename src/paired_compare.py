"""Eşleştirilmiş bootstrap: iki modelin (ya da DSS'in) test macro-F1 farkı anlamlı mı?

Güven aralıklarının çakışması farkın anlamsız olduğunu göstermez: modeller AYNI test satırlarında
değerlendirildiği için doğru karşılaştırma, her bootstrap örnekleminde iki skorun farkını almaktır.
Fark aralığı 0'ı içermiyorsa fark anlamlıdır (%95).

Kullanım:
  python src/paired_compare.py --outputs outputs/dataset_23 --pairs DSS-XGB,DSS-RF,XGB-RF
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import main as dss_main


def macro_f1(y, p, n_classes):
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(2 * tp / np.maximum(denom, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs/dataset_23", help="Bir veri setinin sözleşme klasörü")
    p.add_argument("--pairs", default="DSS-XGB,DSS-RF,XGB-RF")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    env = dss_main.EnvironmentModel(args.outputs)
    df = pd.read_csv(os.path.join(args.outputs, "test_predictions.csv.gz"))
    classes, n_classes = env.classes, len(env.classes)
    y = df["y_true"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy().astype(int)
    names = {n for pair in args.pairs.split(",") for n in pair.split("-")}
    pred = {m: env.probas_from_frame(df, [m])[0].argmax(1) for m in names if m != "DSS"}
    if "DSS" in names:
        pred["DSS"] = env.combine(env.probas_from_frame(df)).argmax(1)

    rng = np.random.default_rng(args.seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.n_boot)]
    print(f"{env.id}: {len(y)} test satırı, {args.n_boot} eşleştirilmiş bootstrap; DSS oylayanlar {env.voters}")
    for m in sorted(pred):
        print(f"  {m:5s} macro-F1 {macro_f1(y, pred[m], n_classes):.4f}")
    for pair in args.pairs.split(","):
        a, b = pair.split("-")
        d = np.array([macro_f1(y[i], pred[a][i], n_classes) - macro_f1(y[i], pred[b][i], n_classes) for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff = macro_f1(y, pred[a], n_classes) - macro_f1(y, pred[b], n_classes)
        print(f"  {a} − {b}: fark {diff:+.4f}, %95 aralık [{lo:+.4f}, {hi:+.4f}] "
              f"-> {'anlamlı' if lo > 0 or hi < 0 else 'anlamlı değil'}")


if __name__ == "__main__":
    main()
