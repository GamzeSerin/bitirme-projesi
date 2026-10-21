"""Açıklama Doğrulama özet grafiği (Edge-IIoTset / dnn_edgeiiot).

xai_faithfulness_dnn_edgeiiot.py çıktılarındaki faithfulness_*.json dosyalarını okur ve tek
figürde iki soruyu yanıtlar:
  Sol:  Açıklama yöntemi gerçek etkiyle ne kadar uyumlu? (Spearman; permütasyon ve drop-one'a karşı)
  Sağ:  LIME aynı satırı farklı tohumla açıklayınca ilk 5 özelliğin ne kadarı korunuyor?
SHAP (TreeExplainer) deterministiktir; kararlılığı tanım gereği 1'dir.

Çalıştırma (eğitim gerektirmez, saniyeler sürer):
  python src/xai_faithfulness_summary_dnn_edgeiiot.py
"""

import argparse
import json
import os

import numpy as np

OUT_DIR = "outputs/dnn_edgeiiot/xai_faithfulness"
RUNS = [  # (dosya etiketi, grafikteki ad)
    ("DT_A_tam", "DT, LIME varsayılan"),
    ("RF_A_tam", "RF, LIME varsayılan"),
    ("RF_A_tam_lime5000", "RF, LIME 5000 örnek"),
    ("RF_A_tam_nodisc", "RF, LIME ayrıklaştırmasız"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=OUT_DIR)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for tag, name in RUNS:
        f = os.path.join(args.dir, f"faithfulness_{tag}.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                runs.append((tag, name, json.load(fh)))
    if not runs:
        raise SystemExit(f"{args.dir} içinde faithfulness_*.json bulunamadı")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    refs = [("Permütasyon", "permütasyona karşı"), ("Drop-one", "drop-one'a karşı")]
    bars = []  # (etiket, [spearman perm, spearman drop], renk)
    seen_shap = set()
    for tag, name, d in runs:
        ra = d["rank_agreement"]
        model = tag.split("_")[0]
        if model not in seen_shap:  # SHAP her model için bir kez (LIME ayarı SHAP'ı değiştirmez)
            seen_shap.add(model)
            bars.append((f"{model}, SHAP", [ra["Permütasyon ~ SHAP"]["spearman"],
                                           ra["Drop-one ~ SHAP"]["spearman"]], "#2a78d6"))
        bars.append((name, [ra["Permütasyon ~ LIME"]["spearman"], ra["Drop-one ~ LIME"]["spearman"]],
                     "#d9822b"))
    y = np.arange(len(bars))
    h = 0.38
    for j, (ref, lab) in enumerate(refs):
        vals = [b[1][j] for b in bars]
        a1.barh(y + (j - 0.5) * h, vals, height=h * 0.92, color=[b[2] for b in bars],
                alpha=1.0 if j == 0 else 0.5, label=lab)
        for yi, v in zip(y + (j - 0.5) * h, vals):
            a1.text(max(v, 0) + 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    a1.set_yticks(y, [b[0] for b in bars])
    a1.invert_yaxis()
    a1.set_xlim(0, 1.05)
    a1.set_xlabel("Spearman sıra uyumu (koyu: permütasyon, açık: drop-one)")
    a1.set_title("Gerçek etkiyle uyum", fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)

    names, means, mins = ["SHAP (deterministik)"], [1.0], [1.0]
    for tag, name, d in runs:
        s = d["lime_stability"]
        names.append(name)
        means.append(s["top5_jaccard_mean"])
        mins.append(s["top5_jaccard_min"])
    y2 = np.arange(len(names))
    a2.barh(y2, means, color=["#2a78d6"] + ["#d9822b"] * (len(names) - 1), height=0.6)
    a2.scatter(mins, y2, color="#14202b", marker="|", s=160, zorder=3)
    for yi, m in zip(y2, means):
        a2.text(m + 0.02, yi, f"{m:.2f}", va="center", fontsize=8)
    a2.set_yticks(y2, names)
    a2.invert_yaxis()
    a2.set_xlim(0, 1.1)
    a2.set_xlabel("İlk 5 özellikte tohumlar arası Jaccard\n(çubuk: ortalama, | : en kötü satır)")
    a2.set_title("Açıklama kararlılığı (30 satır, 3 tohum)", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Edge-IIoTset, RF ve DT: SHAP ve LIME açıklamalarının doğrulanması", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.dir, "fig_xai_faithfulness_summary_dnn_edgeiiot_RF_DT.png")
    fig.savefig(out, dpi=200)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    main()
