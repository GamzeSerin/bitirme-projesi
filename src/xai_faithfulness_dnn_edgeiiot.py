"""Kişi 3 - Açıklama Doğrulama (XAI faithfulness): açıklama yöntemleri modelin gerçek davranışını doğru gösteriyor mu?

Soru: SHAP ve LIME'ın özellik önem sıralamaları, bir özellik modelden gerçekten
çıkarıldığında kaybedilen performansla (ve permütasyon önemiyle) ne kadar örtüşüyor?
Açıklamaların "sadakati" (faithfulness) ve LIME'ın kararlılığı ölçülür.

Yöntem (RF, asıl deneydeki satırlar ve 80/20 bölme, ablation'daki sabit ayarlar):
  1. Referans önemler ("gerçek" etki):
     - drop-one: her özellik tek tek çıkarılıp model yeniden eğitilir; test macro-F1 kaybı
     - permütasyon: test kümesinde özellik karıştırılınca macro-F1 kaybı (yeniden eğitim yok)
  2. Açıklama önemleri:
     - SHAP (TreeExplainer): sınıf başına n test satırı, ortalama |SHAP|
     - LIME (LimeTabularExplainer): sınıf başına m test satırı, tahmin edilen sınıf için |ağırlık|
  3. Karşılaştırma: Spearman / Kendall sıra korelasyonu, ilk-5 ve ilk-10 örtüşmesi (Jaccard),
     sınıf bazında SHAP–LIME ilk-3 uyumu.
  4. LIME kararlılığı: aynı satırlar farklı tohumlarla tekrar açıklanır; ilk-5 Jaccard ve
     ağırlıkların Spearman korelasyonu.
Not: drop-one, birbirinin yerini tutabilen (korelasyonlu) özelliklerin etkisini küçük gösterir;
bu yüzden permütasyon ve ablation'daki grup sonuçlarıyla birlikte yorumlanır.

Çıktı: outputs/dnn_edgeiiot/xai_faithfulness/ (faithfulness_<model>_<koşul>.json + grafik)

Kullanım (Kaggle CPU, ~25-35 dk; GPU gerekmez):
  pip install lime shap
  python src/xai_faithfulness_dnn_edgeiiot.py --data-file "<.../DNN-EdgeIIoT-dataset.csv>"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ablation_dnn_edgeiiot import build_models
from common import contract
from common.taxonomy import COMMON_CLASSES
from model_dnn_edgeiiot import load_data

OUT_DIR = "outputs/dnn_edgeiiot/xai_faithfulness"


def macro_f1(model, X, y):
    return float(f1_score(y, model.predict(X), average="macro"))


def rank_corr(a, b):
    """a, b: {özellik: önem}. Ortak özellikler üzerinde Spearman ve Kendall."""
    feats = sorted(set(a) & set(b))
    x, y = [a[f] for f in feats], [b[f] for f in feats]
    return {"spearman": round(float(spearmanr(x, y).statistic), 3),
            "kendall": round(float(kendalltau(x, y).statistic), 3)}


def topk_jaccard(a, b, k):
    ta = set(sorted(a, key=a.get, reverse=True)[:k])
    tb = set(sorted(b, key=b.get, reverse=True)[:k])
    return round(len(ta & tb) / len(ta | tb), 3)


def normalize(d):
    s = sum(max(v, 0) for v in d.values()) or 1.0
    return {k: max(v, 0) / s for k, v in d.items()}


# ---------------------------------------------------------------- açıklama yöntemleri
def shap_importance(model, Xs, ys, classes):
    import shap
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (n, f, c)
    sv = np.abs(sv)
    feats = list(Xs.columns)
    glob = sv.mean(axis=(0, 2))
    per_class = {}
    for i, c in enumerate(classes):
        rows = ys == i
        imp = sv[rows, :, i].mean(axis=0) if rows.any() else sv[:, :, i].mean(axis=0)
        per_class[c] = dict(zip(feats, map(float, imp)))
    return dict(zip(feats, map(float, glob))), per_class


LIME_DISCRETIZE = True  # --lime-no-discretize ile kapatılır (duyarlılık denemesi)


def lime_explainer(Xtr, classes, seed):
    from lime.lime_tabular import LimeTabularExplainer
    kw = dict(feature_names=list(Xtr.columns), class_names=classes, mode="classification",
              random_state=seed)
    if not LIME_DISCRETIZE:
        return LimeTabularExplainer(Xtr.to_numpy(), discretize_continuous=False, **kw)
    try:
        return LimeTabularExplainer(Xtr.to_numpy(), discretize_continuous=True, **kw)
    except Exception as e:  # çoğu satırı 0 olan sütunlarda çeyrek sınırları çakışabilir
        print(f"  ! LIME ayrıklaştırma başarısız ({e}); discretize_continuous=False ile devam")
        return LimeTabularExplainer(Xtr.to_numpy(), discretize_continuous=False, **kw)


def lime_weights(explainer, model, row, label, num_samples, columns=None):
    # LIME numpy dizisi gönderir; özellik adlarıyla DataFrame'e çevirmek sklearn uyarılarını önler
    predict = (lambda Z: model.predict_proba(pd.DataFrame(Z, columns=columns))) if columns is not None \
        else model.predict_proba
    exp = explainer.explain_instance(row, predict, labels=(label,),
                                     num_features=len(row), num_samples=num_samples)
    w = np.zeros(len(row))
    for j, val in exp.as_map()[label]:
        w[j] = val
    return w


def lime_importance(model, Xtr_bg, Xs, ys, classes, seed, num_samples):
    feats = list(Xs.columns)
    expl = lime_explainer(Xtr_bg, classes, seed)
    pred = model.predict(Xs)
    W = np.zeros((len(Xs), len(feats)))
    for i in range(len(Xs)):
        W[i] = lime_weights(expl, model, Xs.iloc[i].to_numpy(), int(pred[i]), num_samples, feats)
    A = np.abs(W)
    per_class = {}
    for i, c in enumerate(classes):
        rows = ys == i
        per_class[c] = dict(zip(feats, map(float, (A[rows] if rows.any() else A).mean(axis=0))))
    return dict(zip(feats, map(float, A.mean(axis=0)))), per_class


def lime_stability(model, Xtr_bg, Xs, classes, seeds, num_samples, k=5):
    pred = model.predict(Xs)
    runs = []
    for s in seeds:
        expl = lime_explainer(Xtr_bg, classes, s)
        runs.append(np.array([lime_weights(expl, model, Xs.iloc[i].to_numpy(), int(pred[i]), num_samples,
                                           list(Xs.columns))
                              for i in range(len(Xs))]))
    jac, rho = [], []
    for i in range(len(Xs)):
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                wa, wb = np.abs(runs[a][i]), np.abs(runs[b][i])
                ta, tb = set(np.argsort(wa)[-k:]), set(np.argsort(wb)[-k:])
                jac.append(len(ta & tb) / len(ta | tb))
                rho.append(spearmanr(wa, wb).statistic)
    return {"rows": int(len(Xs)), "seeds": list(seeds), "num_samples": num_samples,
            f"top{k}_jaccard_mean": round(float(np.mean(jac)), 3),
            f"top{k}_jaccard_min": round(float(np.min(jac)), 3),
            "weight_spearman_mean": round(float(np.nanmean(rho)), 3)}


# ---------------------------------------------------------------- grafik
def plot(methods, out_path, top=12, title="Edge-IIoTset: açıklama yöntemleri ve gerçek etki"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = {m: normalize(v) for m, v in methods.items()}
    order = sorted(norm["SHAP"], key=lambda f: -max(norm[m].get(f, 0) for m in norm))[:top]
    colors = {"Drop-one": "#14202b", "Permütasyon": "#9aa5b1", "SHAP": "#2a78d6", "LIME": "#d9822b"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.8 / len(norm)
    for j, (m, d) in enumerate(norm.items()):
        ys = np.arange(len(order)) + (j - (len(norm) - 1) / 2) * w
        ax.barh(ys, [d.get(f, 0) for f in order], height=w * 0.92, color=colors.get(m, "#555"), label=m)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Normalize önem (yöntem içinde toplam = 1)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-features", default="", help="Ör. kısayolsuz koşul için 6 sütun")
    p.add_argument("--tag", default="A_tam", help="Çıktı adındaki koşul etiketi")
    p.add_argument("--model", default="RF", choices=["RF", "DT"],
                   help="RF: birincil model (DSS açıklamaları ondan); DT: DSS üçlüsünde, kendi başına okunabilir")
    p.add_argument("--shap-per-class", type=int, default=100)
    p.add_argument("--lime-per-class", type=int, default=50)
    p.add_argument("--lime-samples", type=int, default=1000)
    p.add_argument("--stability-rows", type=int, default=30)
    p.add_argument("--perm-rows", type=int, default=20000)
    p.add_argument("--skip-drop-one", action="store_true")
    p.add_argument("--drop-one-from", default="",
                   help="Drop-one değerlerini önceki bir faithfulness_*.json dosyasından al (yeniden eğitmeden)")
    p.add_argument("--lime-no-discretize", action="store_true",
                   help="LIME'ı discretize_continuous=False ile çalıştır (duyarlılık denemesi)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.shap_per_class, args.lime_per_class = 3000, 20, 5
        args.lime_samples, args.stability_rows, args.perm_rows = 300, 5, 2000
    global LIME_DISCRETIZE
    LIME_DISCRETIZE = not args.lime_no_discretize
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()

    drop = [f for f in args.drop_features.split(",") if f]
    X, fine, category, _ = load_data(args.data_file, args.cap, args.seed, drop, drop_conflicts=True)
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    feats = list(X.columns)
    print(f"Örneklem: {len(y)} satır, {len(feats)} özellik, {len(classes)} kategori ({args.tag})")

    base = build_models(args.seed, args.n_jobs)[args.model]
    model = clone(base).fit(Xtr, ytr)
    f1_full = macro_f1(model, Xte, yte)
    print(f"{args.model} test macro-F1 = {f1_full:.4f}")

    methods = {}
    # 1a. drop-one (yeniden eğitim)
    if args.drop_one_from:
        with open(args.drop_one_from, encoding="utf-8") as f:
            prev = json.load(f)
        methods["Drop-one"] = {k: float(v) for k, v in prev["importances"]["Drop-one"].items() if k in feats}
        print(f"Drop-one önceki çalıştırmadan alındı: {args.drop_one_from}")
    elif not args.skip_drop_one:
        drop_one = {}
        for i, f in enumerate(feats, 1):
            m = clone(base).fit(Xtr.drop(columns=[f]), ytr)
            drop_one[f] = f1_full - macro_f1(m, Xte.drop(columns=[f]), yte)
            print(f"  drop-one {i:2d}/{len(feats)} {f:28s} kayıp={drop_one[f]:+.4f}")
        methods["Drop-one"] = drop_one
    # 1b. permütasyon
    pi_idx = contract.stratified_sample(yte, max(1, args.perm_rows // len(classes)), args.seed)
    pi = permutation_importance(model, Xte.iloc[pi_idx], yte[pi_idx], scoring="f1_macro",
                                n_repeats=3, random_state=args.seed, n_jobs=1)
    methods["Permütasyon"] = dict(zip(feats, map(float, pi.importances_mean)))
    print("Permütasyon önemi tamam")
    # 2a. SHAP
    s_idx = contract.stratified_sample(yte, args.shap_per_class, args.seed)
    methods["SHAP"], shap_cls = shap_importance(model, Xte.iloc[s_idx], yte[s_idx], classes)
    print(f"SHAP tamam ({len(s_idx)} satır)")
    # 2b. LIME
    bg = Xtr.iloc[contract.stratified_sample(ytr, 1000, args.seed)]
    l_idx = contract.stratified_sample(yte, args.lime_per_class, args.seed)
    t0 = time.perf_counter()
    methods["LIME"], lime_cls = lime_importance(model, bg, Xte.iloc[l_idx], yte[l_idx], classes,
                                                args.seed, args.lime_samples)
    print(f"LIME tamam ({len(l_idx)} satır, {time.perf_counter() - t0:.0f} s)")
    stab_idx = contract.stratified_sample(yte, max(1, args.stability_rows // len(classes)), args.seed + 1)
    stability = lime_stability(model, bg, Xte.iloc[stab_idx], classes, [0, 1, 2], args.lime_samples)
    print(f"LIME kararlılığı: {stability}")

    # 3. karşılaştırma
    names = list(methods)
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            comparisons[f"{a} ~ {b}"] = {**rank_corr(methods[a], methods[b]),
                                         "top5_jaccard": topk_jaccard(methods[a], methods[b], 5),
                                         "top10_jaccard": topk_jaccard(methods[a], methods[b], 10)}
    per_class_top3 = {c: {"SHAP": sorted(shap_cls[c], key=shap_cls[c].get, reverse=True)[:3],
                          "LIME": sorted(lime_cls[c], key=lime_cls[c].get, reverse=True)[:3],
                          "top3_jaccard": topk_jaccard(shap_cls[c], lime_cls[c], 3)} for c in classes}
    top10 = {m: [(f, round(v, 5)) for f, v in sorted(d.items(), key=lambda kv: -kv[1])[:10]]
             for m, d in methods.items()}

    out = {"dataset_id": "dnn_edgeiiot", "condition": args.tag, "dropped_features": drop,
           "model": f"{args.model} (ablation sabit ayarları)", "test_macro_f1": round(f1_full, 4),
           "classes": classes, "settings": {k: v for k, v in vars(args).items() if k != "data_file"},
           "top10_by_method": top10, "rank_agreement": comparisons,
           "per_class_shap_vs_lime_top3": per_class_top3, "lime_stability": stability,
           "importances": {m: {f: round(v, 6) for f, v in d.items()} for m, d in methods.items()},
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out_dir, f"faithfulness_{args.model}_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot(methods, os.path.join(args.out_dir, f"fig_xai_faithfulness_dnn_edgeiiot_{args.model}_{args.tag}.png"),
         title=f"Edge-IIoTset, {args.model}: açıklama yöntemleri ve gerçek etki")

    print("\nİlk 5 özellik:")
    for m, lst in top10.items():
        print(f"  {m:12s} {[f for f, _ in lst[:5]]}")
    print("\nSıra uyumu (Spearman | Kendall | ilk-5 Jaccard | ilk-10 Jaccard):")
    for k, v in comparisons.items():
        print(f"  {k:28s} {v['spearman']:+.3f} | {v['kendall']:+.3f} | {v['top5_jaccard']:.2f} | {v['top10_jaccard']:.2f}")
    print("\nSınıf bazında SHAP ve LIME ilk-3 uyumu:")
    for c, v in per_class_top3.items():
        print(f"  {c:11s} J={v['top3_jaccard']:.2f}  SHAP {v['SHAP']}  LIME {v['LIME']}")
    print(f"\nÇıktılar: {args.out_dir}  ({out['runtime_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
