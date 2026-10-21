"""Tohum kontrolü: görülmemiş saldırı sonuçları eğitim rastgeleliğine ne kadar duyarlı? (dnn_edgeiiot)

Veri bölmesi sabittir (ekip protokolü, seed 42); yalnızca modelin eğitim tohumu değişir (42, 1, 2).
CP-TabM her aile için dün iç provanın seçtiği ayarla (kör oranı + skor) eğitilir; yeniden seçim yapılmaz.

Karşılaştırılanlar (koşul C, 5 ailenin her biri):
  TabM · disagree               sade TabM, üye anlaşmazlığı (6 kategori)
  TabM (alt tür) · disagree     sade TabM, 14 alt türle eğitim
  CP-TabM v3 · seçilen          iç provanın seçtiği skor
  CP-TabM · expert_energy_z     "emin olma derecesi" (sessiz BENIGN kaçışı iddiası için)
  kNN                           modelden bağımsız (eğitim tohumu yok, tek değer)

Girdiler:
  tohum 42: outputs/dnn_edgeiiot/open_set/*.json ve cp_tune/cp_tune_C_seq_ack_yok.json (dünkü koşular)
  tohum 1, 2: outputs/dnn_edgeiiot/seeds/s<N>/*.json (open_set_lofo ... --model-seed N)

Çalıştırma:  python src/seed_check_dnn_edgeiiot.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = "outputs/dnn_edgeiiot"
FAMILIES = ["DDOS", "RECON", "WEB", "BRUTEFORCE", "MALWARE"]
COND = "C_seq_ack_yok"
METRICS = ["auroc", "silent_benign@cal", "detect@cal"]


def rows_from_lofo(path, model, score, method, seed, chosen=None):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for r in d["results"]:
        if r["model"] != model:
            continue
        want = chosen.get(r["family"]) if chosen else score
        if r["score"] != want:
            continue
        cf1 = d["families"][r["family"]]["models"].get(model, {}).get("closed_f1_known", np.nan)
        out.append({"method": method, "seed": seed, "family": r["family"], "closed_f1": cf1,
                    **{m: r[m] for m in METRICS}})
    return out


def load(base):
    rows = []
    cp_tune = os.path.join(base, "cp_tune", f"cp_tune_{COND}.json")
    chosen = {}
    if os.path.exists(cp_tune):
        with open(cp_tune, encoding="utf-8") as f:
            t = json.load(f)
        chosen = {fam: v["chosen"]["score"] for fam, v in t["families"].items()}
        for fam, v in t["families"].items():   # tohum 42, CP-TabM v3
            allsc = v["outer_all_scores_not_used_for_selection"]
            cf1 = v["outer_selected"]["closed_f1_known"]
            for method, sc in [("CP-TabM v3 · seçilen", chosen[fam]),
                               ("CP-TabM · emin olma derecesi", "expert_energy_z")]:
                rows.append({"method": method, "seed": 42, "family": fam, "closed_f1": cf1,
                             **{m: allsc[sc][m] for m in METRICS}})
    o = os.path.join(base, "open_set")
    rows += rows_from_lofo(os.path.join(o, f"open_set_lofo_TabM_{COND}.json"), "TabM", "disagree",
                           "TabM · anlaşmazlık", 42)
    rows += rows_from_lofo(os.path.join(o, f"open_set_lofo_TabM_{COND}_fine.json"), "TabM", "disagree",
                           "TabM (alt tür) · anlaşmazlık", 42)
    rows += rows_from_lofo(os.path.join(o, f"open_set_lofo_TabM_{COND}.json"), "bağımsız", "knn", "kNN", 42)
    for s in [1, 2, 3, 4]:
        d = os.path.join(base, "seeds", f"s{s}")
        if not os.path.isdir(d):
            continue
        rows += rows_from_lofo(os.path.join(d, f"open_set_lofo_TabM_{COND}.json"), "TabM", "disagree",
                               "TabM · anlaşmazlık", s)
        rows += rows_from_lofo(os.path.join(d, f"open_set_lofo_TabM_{COND}_fine.json"), "TabM", "disagree",
                               "TabM (alt tür) · anlaşmazlık", s)
        cp = os.path.join(d, f"open_set_lofo_CPTabM_{COND}.json")
        rows += rows_from_lofo(cp, "CPTabM", None, "CP-TabM v3 · seçilen", s, chosen=chosen)
        rows += rows_from_lofo(cp, "CPTabM", "expert_energy_z", "CP-TabM · emin olma derecesi", s)
    return pd.DataFrame(rows)


def plot(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["kNN", "TabM · anlaşmazlık", "TabM (alt tür) · anlaşmazlık", "CP-TabM v3 · seçilen",
             "CP-TabM · emin olma derecesi"]
    colors = {"kNN": "#d9822b", "TabM · anlaşmazlık": "#2a78d6", "TabM (alt tür) · anlaşmazlık": "#7fb0e6",
              "CP-TabM v3 · seçilen": "#1f9e6e", "CP-TabM · emin olma derecesi": "#8fd1b5"}
    per_seed = df.groupby(["method", "seed"])[["auroc", "silent_benign@cal"]].mean().reset_index()
    methods = [m for m in order if m in set(per_seed["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, lab in [(axes[0], "auroc", "AUROC, 5 ailenin ortalaması"),
                         (axes[1], "silent_benign@cal", "Sessizce BENIGN'e kaçan bilinmeyen oranı")]:
        for i, m in enumerate(methods):
            v = per_seed[per_seed.method == m][col].to_numpy()
            ax.bar(i, v.mean(), color=colors[m], width=0.65)
            if len(v) > 1:
                ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="#14202b", capsize=4, lw=1)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="#14202b", s=10, zorder=3)
            ax.text(i, v.mean() + (0.01 if col == "auroc" else 0.002),
                    f"{v.mean():.3f}" + (f"\n±{v.std(ddof=1):.3f}" if len(v) > 1 else ""),
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(methods)), [m.replace(" · ", "\n") for m in methods], fontsize=8)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.6, 0.95)
    n_seeds = per_seed.groupby("method")["seed"].nunique().max()
    fig.suptitle(f"Edge-IIoTset, koşul C: eğitim tohumuna duyarlılık ({n_seeds} tohum; noktalar = tek tohum, "
                 "çubuk = ortalama ± std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE)
    args = p.parse_args()
    df = load(args.base)
    if df.empty:
        raise SystemExit("Sonuç bulunamadı")
    out_dir = os.path.join(args.base, "seeds")
    os.makedirs(out_dir, exist_ok=True)
    fam = (df.groupby(["method", "family"])[["auroc", "silent_benign@cal", "closed_f1"]]
           .agg(["mean", "std"]).round(4))
    per_seed = df.groupby(["method", "seed"])[["auroc", "silent_benign@cal", "detect@cal", "closed_f1"]].mean()
    overall = per_seed.groupby("method").agg(["mean", "std", "count"]).round(4)
    print("Aile bazında (tohumlar arası ortalama ± std):")
    print(fam.to_string())
    print("\n5 ailenin ortalaması (tohumlar arası ortalama ± std):")
    print(overall.to_string())
    res = {"condition": COND, "note": "veri bölmesi sabit (seed 42); yalnızca model eğitim tohumu değişir",
           "rows": df.to_dict(orient="records"),
           "overall": {m: {c: {k: float(v) for k, v in overall.loc[m, c].items()} for c in
                           ["auroc", "silent_benign@cal", "detect@cal", "closed_f1"]} for m in overall.index}}
    with open(os.path.join(out_dir, f"seed_check_{COND}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    plot(df, os.path.join(out_dir, f"fig_seed_check_dnn_edgeiiot_{COND}.png"))
    print(f"\nÇıktılar: {out_dir}")


if __name__ == "__main__":
    main()
