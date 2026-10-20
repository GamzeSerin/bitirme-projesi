"""Görülmemiş saldırı deneyi özet grafiği (Edge-IIoTset / dnn_edgeiiot).

open_set_lofo_dnn_edgeiiot.py çıktılarındaki open_set_lofo_*.json dosyalarını okur (RF, TabM,
ileride CP-TabM; hangileri varsa) ve koşullar (A, C, E) arasında ailelerin ortalamasını karşılaştırır:
  Sol:  AUROC (görülmemiş aileyi bilinenlerden ayırma)
  Sağ:  sessizce BENIGN'e kaçan bilinmeyen oranı (kalibrasyon eşiğinde; en tehlikeli hata)

Çalıştırma (eğitim gerektirmez):  python src/open_set_summary_dnn_edgeiiot.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "outputs/dnn_edgeiiot/open_set"
CONDITIONS = [("A_tam", "A: tüm özellikler"), ("C_seq_ack_yok", "C: seq/ack yok"),
              ("E_hepsi_cikarildi", "E: kısayolsuz")]
COLORS = {"msp": "#9aa5b1", "entropy": "#5f6b7a", "disagree": "#2a78d6", "energy": "#7b4fd6",
          "knn": "#d9822b"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=OUT_DIR)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, closed = [], []
    for f in sorted(glob.glob(os.path.join(args.dir, "open_set_lofo_*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for r in d["results"]:
            rows.append({**r, "condition": d["condition"]})
        for fam, v in d["families"].items():
            for m, mv in v["models"].items():
                closed.append({"condition": d["condition"], "model": m, "family": fam,
                               "closed_f1": mv["closed_f1_known"]})
    if not rows:
        raise SystemExit(f"{args.dir} içinde open_set_lofo_*.json yok")
    df = pd.DataFrame(rows).drop_duplicates(["condition", "family", "model", "score"])
    # Eski çalıştırmalarda RF'nin saf yapraklı turlarında "disagree" yazılmamıştı; o durumda tanım gereği
    # entropiyle aynıdır, ortalamalar 5 aile üzerinden kalsın diye entropi satırı kopyalanır.
    have = set(map(tuple, df.loc[df.score == "disagree", ["condition", "family", "model"]].values))
    fill = df[(df.score == "entropy") & (df.model != "bağımsız")]
    fill = fill[[tuple(r) not in have for r in fill[["condition", "family", "model"]].values]]
    df = pd.concat([df, fill.assign(score="disagree")], ignore_index=True)
    df["yöntem"] = df["model"] + " · " + df["score"]
    mean = df.groupby(["condition", "yöntem", "score"])[["auroc", "silent_benign@cal"]].mean().reset_index()
    cf1 = pd.DataFrame(closed).groupby(["condition", "model"])["closed_f1"].mean()

    conds = [c for c in CONDITIONS if c[0] in set(mean["condition"])]
    # Okunabilirlik için entropi gösterilmez (her iki modelde msp ile neredeyse aynı); tablo JSON/CSV'de
    order = ["RF · msp", "RF · disagree", "TabM · msp", "TabM · energy", "TabM · disagree",
             "CP-TabM · signature", "bağımsız · knn"]
    present = set(mean["yöntem"])
    methods = [m for m in order if m in present] + sorted(
        m for m in present if m not in order and not m.endswith("entropy"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    w = 0.8 / len(methods)
    for ax, col, lab in [(axes[0], "auroc", "AUROC (0,5 = rastgele)"),
                         (axes[1], "silent_benign@cal", "Sessizce BENIGN'e kaçan bilinmeyen oranı")]:
        for j, m in enumerate(methods):
            sc = m.split(" · ")[1]
            vals = [mean.query("condition == @c and yöntem == @m")[col].mean() for c, _ in conds]
            xs = np.arange(len(conds)) + (j - (len(methods) - 1) / 2) * w
            hatch = "//" if m.startswith("TabM") else None
            ax.bar(xs, vals, width=w * 0.92, color=COLORS.get(sc, "#555"), hatch=hatch,
                   edgecolor="white", label=m)
            for x, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(x, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                            rotation=90 if len(methods) > 5 else 0)
        labels = []
        for c, name in conds:
            f1s = "\n".join(f"bilinen F1 {m}: {v:.2f}" for (cc, m), v in cf1.items() if cc == c)
            labels.append(f"{name}\n{f1s}")
        ax.set_xticks(range(len(conds)), labels, fontsize=8)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.3, 1.1)
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.12)
    axes[0].axhline(0.5, color="#14202b", lw=0.8, ls="--")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=len(methods), loc="upper center",
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Edge-IIoTset: görülmemiş saldırı ailesi tespiti, 5 ailenin ortalaması", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    models = "_".join(sorted(set(m for m in df["model"] if m != "bağımsız")))
    out = os.path.join(args.dir, f"fig_open_set_summary_dnn_edgeiiot_{models}.png")
    fig.savefig(out, dpi=200)
    print("Kaydedildi:", out)
    print(mean.pivot_table(index="yöntem", columns="condition", values=["auroc", "silent_benign@cal"]).round(3))


if __name__ == "__main__":
    main()
