"""CP-TabM / FP-TabM ayar seçimi: iç içe LOFO ("iç prova"), Edge-IIoTset / dnn_edgeiiot.

Amaç: kör üye oranını ve bilinmeyen skorunu TEST AİLESİNE HİÇ BAKMADAN seçmek, sonra seçilen ayarı
dış testte bir kez çalıştırmak. Böylece "sonuca bakıp ayar yapma" (test sızıntısı) olmaz.

Protokol (her dış aile U için):
  1. Dış bölme open_set_lofo ile aynıdır (ekip 80/20, seed 42). U eğitimden tamamen çıkarılır.
  2. İç prova: U'suz eğitim verisi kendi içinde %80 / %20 bölünür. Kalan saldırı ailelerinin her biri
     (V) sırayla "sahte bilinmeyen" yapılır: model V'siz iç eğitimle eğitilir, iç testte V'yi ayırma
     başarısı ölçülür. Hız için iç eğitim kategori başına --inner-cap satırla sınırlanır.
  3. Seçim ölçütü (önceden belirlendi): iç ailelerin ortalama AUROC'u en yüksek (kör oranı, skor)
     çifti; ortalaması en iyiye 0,005'ten yakın olanlar arasında sessiz BENIGN kaçışı en düşük olan.
  4. Seçilen kör oranıyla model U'suz TÜM eğitim verisiyle yeniden eğitilir ve dış testte bir kez
     değerlendirilir. Raporlanan ana sonuç seçilen skordur; diğer skorlar yalnızca şeffaflık için
     kaydedilir ve seçimde kullanılmaz.

Çalıştırma (GPU önerilir, ~1 saat):
  python src/cp_tune_dnn_edgeiiot.py --data-file <csv>                 # koşul C, 3 kör oranı
  python src/cp_tune_dnn_edgeiiot.py --data-file <csv> --quick         # tesisat denemesi (~2 dk)
  python src/cp_tune_dnn_edgeiiot.py --data-file <csv> --model FPTabM  # özellik-bölümlü üyeler
  ... --label-level fine                                                # alt türlerle eğitim
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ablation_dnn_edgeiiot import SCENARIOS
from common.taxonomy import COMMON_CLASSES
from common.taxonomy import edgeiiot_category
from model_dnn_edgeiiot import load_data
from open_set_lofo_dnn_edgeiiot import FAMILIES, evaluate, model_scores

OUT_DIR = "outputs/dnn_edgeiiot/cp_tune"
SCORES_BY_MODEL = {
    "CPTabM": ["signature", "expert_disagree", "expert_energy_z", "comb_max", "comb_mean"],
    "FPTabM": ["msp", "entropy", "disagree", "energy"],
}
SCORES = SCORES_BY_MODEL["CPTabM"]


def cap_per_category(idx, category, cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(category[idx]):
        ix = idx[category[idx] == c]
        keep.append(rng.choice(ix, size=min(cap, len(ix)), replace=False))
    return np.sort(np.concatenate(keep))


def run_once(X, fine, category, fit_pool, test_idx, unknown_fam, value, args, epochs):
    """fit_pool'dan (unknown_fam hariç) eğit, test_idx'te unknown_fam'i ayırmayı ölç.
    value: CPTabM için kör üye oranı, FPTabM için üye başına özellik oranı."""
    present = [c for c in COMMON_CLASSES if c in set(category[fit_pool])]
    known = [c for c in present if c != unknown_fam]
    tr = fit_pool[category[fit_pool] != unknown_fam]
    tr_fit, tr_cal = train_test_split(tr, test_size=0.1, stratify=fine[tr], random_state=args.seed)
    if args.label_level == "fine":
        labels = sorted(set(fine[tr_fit]))
        yfit = np.array([labels.index(c) for c in fine[tr_fit]])
        label_cat = np.array([edgeiiot_category(c) for c in labels])
    else:
        labels = known
        yfit = np.array([known.index(c) for c in category[tr_fit]])
        label_cat = np.array(known)
    if args.model == "CPTabM":
        from cp_tabm_classifier import CPTabMClassifier
        model = CPTabMClassifier(blind_frac=value, lam_energy=args.cp_lam, m_in=args.cp_m_in,
                                 m_out=args.cp_m_out, max_epochs=epochs, random_state=args.seed,
                                 verbose=False, feature_frac=args.cp_feature_frac)
    else:
        from tabm_classifier import TabMClassifier
        model = TabMClassifier(max_epochs=epochs, random_state=args.seed, verbose=False,
                               feature_frac=value)
    t0 = time.perf_counter()
    model.fit(X.iloc[tr_fit], yfit)
    fit_s = time.perf_counter() - t0
    _, S_cal = model_scores(args.model, model, X.iloc[tr_cal])
    P, S_te = model_scores(args.model, model, X.iloc[test_idx])
    pred = label_cat[P.argmax(1)]
    is_unk = category[test_idx] == unknown_fam
    names = category[test_idx]
    closed = float(f1_score(names[~is_unk], pred[~is_unk], average="macro"))
    res = {s: evaluate(S_cal[s], S_te[s], is_unk, pred, names, "BENIGN", args.target) for s in SCORES}
    return model, labels, res, closed, fit_s, is_unk


def choose(inner_rows):
    df = pd.DataFrame(inner_rows)
    g = df.groupby(["blind_frac", "score"])[["auroc", "silent_benign@cal"]].mean().reset_index()
    best = g["auroc"].max()
    near = g[g["auroc"] >= best - 0.005].sort_values(["silent_benign@cal", "auroc"], ascending=[True, False])
    pick = near.iloc[0]
    return float(pick["blind_frac"]), str(pick["score"]), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--condition", default="C_seq_ack_yok", choices=list(SCENARIOS))
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--model", default="CPTabM", choices=list(SCORES_BY_MODEL),
                   help="CPTabM: sınıf-bölümlü (ızgara = kör oranı); FPTabM: özellik-bölümlü (ızgara = özellik oranı)")
    p.add_argument("--blind-fracs", default=None, help="Izgara; varsayılan CPTabM 0.25,0.125,0.0625 / FPTabM 0.3,0.5,0.7")
    p.add_argument("--label-level", default="category", choices=["category", "fine"])
    p.add_argument("--cp-feature-frac", type=float, default=None)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--inner-cap", type=int, default=12000, help="İç provada kategori başına eğitim satırı")
    p.add_argument("--inner-epochs", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--cp-lam", type=float, default=0.1)
    p.add_argument("--cp-m-in", type=float, default=-6.0)
    p.add_argument("--cp-m-out", type=float, default=-1.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.inner_cap, args.inner_epochs, args.epochs = 3000, 800, 2, 2
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()
    global SCORES
    SCORES = SCORES_BY_MODEL[args.model]
    if args.blind_fracs is None:
        args.blind_fracs = "0.25,0.125,0.0625" if args.model == "CPTabM" else "0.3,0.5,0.7"
    fracs = [float(f) for f in args.blind_fracs.split(",")]

    X, fine, category, _ = load_data(args.data_file, args.cap, args.seed, drop_conflicts=True)
    drop = [c for c in SCENARIOS[args.condition] if c in X.columns]
    X = X.drop(columns=drop)
    present = [c for c in COMMON_CLASSES if c in set(category)]
    families = [f for f in args.families.split(",") if f in present and f != "BENIGN"]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"Örneklem: {len(category)} satır, {X.shape[1]} özellik, koşul {args.condition}; "
          f"kör oranları {fracs}; skorlar {SCORES}")

    out = {"dataset_id": "dnn_edgeiiot", "condition": args.condition, "dropped_features": drop,
           "protocol": "iç içe LOFO; seçim yalnızca iç provayla (dış test ailesi kullanılmadı)",
           "selection_rule": "iç ailelerin ort. AUROC'u en yüksek; 0,005 içindekilerden sessiz BENIGN kaçışı en düşük",
           "settings": {k: v for k, v in vars(args).items() if k != "data_file"}, "families": {}}
    final_rows = []
    for U in families:
        tU = time.perf_counter()
        pool = idx_tr[category[idx_tr] != U]
        in_tr, in_te = train_test_split(pool, test_size=0.2, stratify=fine[pool], random_state=args.seed)
        in_tr = cap_per_category(in_tr, category, args.inner_cap, args.seed)
        inner_fams = [f for f in families if f != U]
        print(f"\n=== Dış bilinmeyen: {U}  — iç prova ({len(inner_fams)} sahte bilinmeyen x {len(fracs)} kör oranı) ===")
        inner_rows = []
        for bf in fracs:
            for V in inner_fams:
                _, _, res, closed, fit_s, _ = run_once(X, fine, category, in_tr, in_te, V, bf, args,
                                                       args.inner_epochs)
                for s, m in res.items():
                    inner_rows.append({"blind_frac": bf, "inner_unknown": V, "score": s,
                                       "closed_f1": round(closed, 4), **m})
                print(f"  {'kör' if args.model == 'CPTabM' else 'özellik'} {bf:<6} sahte bilinmeyen {V:10s} ({fit_s:.0f} s)  "
                      + "  ".join(f"{s} {res[s]['auroc']:.3f}" for s in SCORES))
        bf, sc, table = choose(inner_rows)
        print(f"  → SEÇİLEN: {'kör' if args.model == 'CPTabM' else 'özellik'} oranı {bf}, skor {sc}  (iç ort. AUROC "
              f"{table.query('blind_frac == @bf and score == @sc')['auroc'].iloc[0]:.3f})")

        model, known, res, closed, fit_s, is_unk = run_once(X, fine, category, idx_tr, idx_te, U, bf, args,
                                                            args.epochs)
        sel = res[sc]
        print(f"  DIŞ TEST ({U}): AUROC {sel['auroc']:.3f}  FPR95 {sel['fpr95']:.3f}  yakalanan "
              f"{sel['detect@cal']:.3f}  BENIGN'e kaçan {sel['silent_benign@cal']:.3f}  bilinen F1 {closed:.4f} "
              f"({fit_s:.0f} s)  [{time.perf_counter() - tU:.0f} s]")
        final_rows.append({"family": U, "blind_frac": bf, "score": sc, "closed_f1": round(closed, 4), **sel})
        ex_idx = idx_te[np.flatnonzero(is_unk)[:5]]
        out["families"][U] = {
            "chosen": {"blind_frac": bf, "score": sc},
            "inner_mean": table.round(4).to_dict(orient="records"),
            "inner_runs": inner_rows,
            "outer_selected": {**sel, "closed_f1_known": round(closed, 4)},
            "outer_all_scores_not_used_for_selection": res,
            "explain_unknown_examples": (model.explain(X.iloc[ex_idx], known)
                                         if args.model == "CPTabM" else None),
        }

    df = pd.DataFrame(final_rows)
    out["final"] = final_rows
    out["final_mean"] = df[["auroc", "fpr95", "detect@cal", "silent_benign@cal", "open_f1@cal",
                            "closed_f1"]].mean().round(4).to_dict()
    out["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    tag = f"{args.model}_{args.condition}" + ("_fine" if args.label_level == "fine" else "")
    with open(os.path.join(args.out_dir, f"tune_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSeçilen ayarlarla dış test ({args.model}, iç provayla seçildi):")
    print(df[["family", "blind_frac", "score", "auroc", "fpr95", "detect@cal", "silent_benign@cal",
              "closed_f1"]].to_string(index=False))
    print("Ortalama:", out["final_mean"])
    print(f"\nÇıktılar: {args.out_dir}  ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
