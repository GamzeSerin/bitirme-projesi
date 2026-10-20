"""Görülmemiş saldırı deneyi, temel skorlar (Edge-IIoTset / dnn_edgeiiot).

Soru: Bir saldırı ailesi eğitimde HİÇ görülmediyse, model onu "bilinmeyen" olarak işaretleyebilir mi,
yoksa sessizce bilinen bir sınıfa (en kötüsü BENIGN) mı atar?

Protokol (leave-one-family-out, LOFO):
  - Ekip bölmesi: tüm temiz veri 80/20, ince etikete göre katmanlı, seed 42 (asıl deneyle aynı test).
  - Her turda bir saldırı ailesi U (DDOS, RECON, WEB, BRUTEFORCE, MALWARE) eğitimden tamamen çıkarılır.
  - Kalan eğitim verisi %90 eğitim / %10 kalibrasyon. Eşik YALNIZCA kalibrasyondaki bilinen satırlarla
    seçilir (bilinenlerin %95'i kabul edilir); bilinmeyen aileden hiçbir bilgi kullanılmaz.
  - Test: asıl test kümesinin tamamı; U'nun satırları "bilinmeyen", diğerleri "bilinen".

Bilinmeyen skorları (büyük = daha "bilinmeyen"):
  msp       1 - en yüksek olasılık                         (RF, TabM)
  entropy   tahmin dağılımının entropisi                    (RF, TabM)
  disagree  topluluk üyeleri arası anlaşmazlık              (RF: ağaçlar, TabM: k üye; yapraklar saf
            olduğunda entropiyle aynıdır)
            = karşılıklı bilgi H(ort. p) - ort. H(p_m)
  energy    -logsumexp(ort. logit)                          (yalnızca TabM)
  knn       eğitimdeki en yakın 5 komşuya ortalama uzaklık  (modelden bağımsız, QuantileTransformer uzayı)

Ölçütler (aile x model x skor):
  auroc, aupr            bilinen/bilinmeyen ayrımı (eşikten bağımsız)
  fpr95                  bilinenlerin %95'i kabul edilirken kabul edilen bilinmeyen oranı (test eşiği; literatür ölçütü)
  detect@cal             kalibrasyon eşiğinde yakalanan bilinmeyen oranı (gerçekçi, sızıntısız)
  known_reject@cal       aynı eşikte yanlışlıkla reddedilen bilinen oranı
  silent_benign@cal      bilinmeyen saldırının kabul edilip BENIGN denmesi (en tehlikeli hata)
  open_f1@cal            K+1 sınıflı (bilinenler + UNKNOWN) macro-F1
Ayrıca her ailede: bilinen testte kapalı-küme macro-F1 ve bilinmeyenlerin hangi sınıfa atandığı.

Kısayol alanlar bilinmeyen tespitini de etkileyebileceği için deney iki koşulda çalıştırılmalı:
  --condition A_tam               (tüm 34 özellik)
  --condition E_hepsi_cikarildi   (port, seq/ack ve şüpheli kimlik alanları çıkarılmış)

Çalıştırma:
  python src/open_set_lofo_dnn_edgeiiot.py --data-file <csv> --models RF            # CPU, ~10 dk
  python src/open_set_lofo_dnn_edgeiiot.py --data-file <csv> --models RF,TabM       # GPU önerilir
  python src/open_set_lofo_dnn_edgeiiot.py --data-file <csv> --condition E_hepsi_cikarildi --models RF,TabM
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from ablation_dnn_edgeiiot import SCENARIOS, build_models
from common.taxonomy import COMMON_CLASSES
from model_dnn_edgeiiot import load_data

OUT_DIR = "outputs/dnn_edgeiiot/open_set"
FAMILIES = ["DDOS", "RECON", "WEB", "BRUTEFORCE", "MALWARE"]
UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------- skorlar
def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def scores_from_members(P_members, logits=None):
    """P_members: (n, m, C) üye olasılıkları. Dönen skorlar: büyük = bilinmeyen."""
    P = P_members.mean(axis=1)
    out = {"msp": 1.0 - P.max(axis=1), "entropy": entropy(P)}
    # Saf yapraklı ağaçlarda üye entropisi 0 olur ve anlaşmazlık entropiyle birebir aynıdır;
    # aileler arası ortalamaların karşılaştırılabilir kalması için yine de her zaman raporlanır.
    out["disagree"] = out["entropy"] - entropy(P_members).mean(axis=1)
    if logits is not None:
        out["energy"] = -logsumexp(logits.mean(axis=1), axis=-1)
    return P, out


def rf_members(model, X, batch=20000):
    """RF: (n, ağaç, C). Her ağaç sınıfları model.classes_ sırasıyla verir."""
    Xn = np.asarray(X, dtype=np.float32)
    parts = []
    for i in range(0, len(Xn), batch):
        parts.append(np.stack([t.predict_proba(Xn[i:i + batch]) for t in model.estimators_], axis=1))
    return np.concatenate(parts)


def model_scores(name, model, X):
    if name.startswith("CPTabM"):
        return model.unknown_scores(X)
    if name == "TabM":
        Xt = model._prep(X)
        logits = model._logits(Xt)
        z = logits - logits.max(-1, keepdims=True)
        Pm = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        return scores_from_members(Pm, logits)
    return scores_from_members(rf_members(model, X))


class KNNScore:
    """Modelden bağımsız uzaklık skoru; yalnızca bilinen eğitim verisine uydurulur."""

    def __init__(self, k=5, max_ref=50000, seed=42):
        self.k, self.max_ref, self.seed = k, max_ref, seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        ref = X[rng.choice(len(X), size=min(self.max_ref, len(X)), replace=False)]
        self.qt_ = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                                       subsample=10**9, random_state=self.seed).fit(ref)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.qt_.transform(ref))
        return self

    def score(self, X, batch=20000):
        X = self.qt_.transform(np.asarray(X, dtype=np.float64))
        return np.concatenate([self.nn_.kneighbors(X[i:i + batch])[0].mean(axis=1)
                               for i in range(0, len(X), batch)])


# ---------------------------------------------------------------- ölçütler
def evaluate(score_cal_known, score_test, is_unknown, pred_test, y_test_names, benign, target=0.95):
    s_k, s_u = score_test[~is_unknown], score_test[is_unknown]
    lab = is_unknown.astype(int)
    tau_cal = float(np.quantile(score_cal_known, target))       # kalibrasyondaki bilinenlerin %95'i kabul
    tau_test = float(np.quantile(s_k, target))                  # literatürdeki FPR95 tanımı
    reject = score_test > tau_cal
    open_pred = np.where(reject, UNKNOWN, pred_test)
    open_true = np.where(is_unknown, UNKNOWN, y_test_names)
    labels = sorted(set(open_true))
    return {
        "auroc": round(float(roc_auc_score(lab, score_test)), 4),
        "aupr": round(float(average_precision_score(lab, score_test)), 4),
        "fpr95": round(float((s_u <= tau_test).mean()), 4),
        "detect@cal": round(float((s_u > tau_cal).mean()), 4),
        "known_reject@cal": round(float((s_k > tau_cal).mean()), 4),
        "silent_benign@cal": round(float(((s_u <= tau_cal) & (pred_test[is_unknown] == benign)).mean()), 4),
        "open_f1@cal": round(float(f1_score(open_true, open_pred, labels=labels, average="macro")), 4),
    }


# ---------------------------------------------------------------- grafik
def plot(rows, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    df["yöntem"] = df["model"] + " · " + df["score"]
    piv = df.pivot_table(index="yöntem", columns="family", values="auroc")
    piv = piv[[f for f in FAMILIES if f in piv.columns]]
    piv["ORT."] = piv.mean(axis=1)
    piv = piv.sort_values("ORT.", ascending=False)
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 2.5, 0.42 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.6 < v < 0.95 else "white")
    ax.axvline(piv.shape[1] - 1.5, color="white", lw=2)
    ax.set_xlabel("Eğitimden çıkarılan (bilinmeyen) aile")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="AUROC (0,5 = rastgele)")
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
    p.add_argument("--condition", default="A_tam", choices=list(SCENARIOS))
    p.add_argument("--models", default="RF",
                   help="Virgülle: RF, TabM, CPTabM (önerilen), CPTabM-noE (enerji kaybı kapalı bileşen testi)")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95, help="Kabul edilecek bilinen oranı")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--cp-blind-frac", type=float, default=0.25, help="CP-TabM: sınıf başına kör üye oranı")
    p.add_argument("--cp-lam", type=float, default=0.1, help="CP-TabM: enerji sınırı kaybının ağırlığı")
    p.add_argument("--cp-m-in", type=float, default=-6.0)
    p.add_argument("--cp-m-out", type=float, default=-1.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap = 3000
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()

    drop = list(SCENARIOS[args.condition])
    # Ablation'daki gibi: satırlar tam koşulda seçilir, sütunlar sonra çıkarılır. Böylece E koşulunda da
    # eğitim/test satırları A ile birebir aynı kalır (yalnızca modelin gördüğü sütunlar değişir).
    X, fine, category, _ = load_data(args.data_file, args.cap, args.seed, drop_conflicts=True)
    drop = [c for c in drop if c in X.columns]
    X = X.drop(columns=drop)
    present = [c for c in COMMON_CLASSES if c in set(category)]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"Örneklem: {len(category)} satır, {X.shape[1]} özellik, kategoriler {present} "
          f"(koşul {args.condition})")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    families = [f for f in args.families.split(",") if f in present and f != "BENIGN"]

    rows, details = [], {}
    for fam in families:
        t0 = time.perf_counter()
        known = [c for c in present if c != fam]
        tr = idx_tr[category[idx_tr] != fam]
        tr_fit, tr_cal = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr],
                                          random_state=args.seed)
        Xfit, Xcal, Xte = X.iloc[tr_fit], X.iloc[tr_cal], X.iloc[idx_te]
        yfit = np.array([known.index(c) for c in category[tr_fit]])
        is_unk = category[idx_te] == fam
        y_te_names = category[idx_te]
        print(f"\n=== Bilinmeyen aile: {fam}  (eğitim {len(tr_fit)}, kalibrasyon {len(tr_cal)}, "
              f"test bilinen {int((~is_unk).sum())}, bilinmeyen {int(is_unk.sum())}) ===")
        details[fam] = {"known_classes": known, "n_fit": int(len(tr_fit)), "n_cal": int(len(tr_cal)),
                        "n_test_known": int((~is_unk).sum()), "n_test_unknown": int(is_unk.sum()),
                        "unknown_fine_labels": sorted(set(fine[idx_te][is_unk])), "models": {}}

        knn = KNNScore(seed=args.seed).fit(Xfit)
        knn_cal, knn_te = knn.score(Xcal), knn.score(Xte)

        for name in models:
            if name == "TabM":
                from tabm_classifier import TabMClassifier
                model = TabMClassifier(max_epochs=3 if args.quick else 60, random_state=args.seed,
                                       verbose=False)
            elif name.startswith("CPTabM"):
                from cp_tabm_classifier import CPTabMClassifier
                model = CPTabMClassifier(blind_frac=args.cp_blind_frac,
                                         lam_energy=0.0 if name == "CPTabM-noE" else args.cp_lam,
                                         m_in=args.cp_m_in, m_out=args.cp_m_out,
                                         max_epochs=3 if args.quick else 60, random_state=args.seed,
                                         verbose=False)
            else:
                model = clone(build_models(args.seed, args.n_jobs)[name])
            tm = time.perf_counter()
            model.fit(Xfit, yfit)
            fit_s = time.perf_counter() - tm
            P_cal, S_cal = model_scores(name, model, Xcal)
            P_te, S_te = model_scores(name, model, Xte)
            pred = np.array(known)[P_te.argmax(1)]
            closed_f1 = f1_score(y_te_names[~is_unk], pred[~is_unk], average="macro")
            assigned = pd.Series(pred[is_unk]).value_counts(normalize=True).round(4).to_dict()
            details[fam]["models"][name] = {"closed_f1_known": round(float(closed_f1), 4),
                                            "fit_seconds": round(fit_s, 1),
                                            "unknown_assigned_to": assigned}
            if name == "TabM" or name.startswith("CPTabM"):
                details[fam]["models"][name]["best_epoch"] = int(model.best_epoch_)
            if name.startswith("CPTabM"):
                info = details[fam]["models"][name]
                info["blind_matrix"] = model.A_.astype(int).tolist()
                info["energy_accept_median"] = round(float(np.median(model.e_acc_)), 3)
                info["energy_reject_median"] = round(float(np.median(model.e_rej_)), 3)
                info["energy_gap_median"] = round(model.energy_gap_, 3)
                # Açıklama örnekleri: görülmemiş aileden 5 satır (XAI bölümü için)
                ex_idx = np.flatnonzero(is_unk)[:5]
                info["explain_unknown_examples"] = model.explain(Xte.iloc[ex_idx], known)
                print(f"    enerji kabul/red medyanı {info['energy_accept_median']} / "
                      f"{info['energy_reject_median']} (ayrım {info['energy_gap_median']})")
            print(f"  {name}: bilinen testte macro-F1 {closed_f1:.4f} ({fit_s:.0f} s); "
                  f"bilinmeyenler şuna atandı: {assigned}")
            S_cal = dict(S_cal, knn=knn_cal)
            S_te = dict(S_te, knn=knn_te)
            for sname in S_te:
                if sname == "knn" and name != models[0]:
                    continue  # model bağımsız; bir kez raporla
                m = evaluate(S_cal[sname][np.isin(category[tr_cal], known)], S_te[sname], is_unk,
                             pred, y_te_names, "BENIGN", args.target)
                row = {"family": fam, "model": "bağımsız" if sname == "knn" else name, "score": sname, **m}
                rows.append(row)
                print(f"    {row['model']:8s} {sname:9s} AUROC {m['auroc']:.3f}  FPR95 {m['fpr95']:.3f}  "
                      f"yakalanan {m['detect@cal']:.3f}  BENIGN'e kaçan {m['silent_benign@cal']:.3f}  "
                      f"open-F1 {m['open_f1@cal']:.3f}")
        print(f"  ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    tag = f"{'_'.join(models)}_{args.condition}"
    df.to_csv(os.path.join(args.out_dir, f"open_set_lofo_{tag}.csv"), index=False)
    summary = (df.groupby(["model", "score"])[["auroc", "fpr95", "detect@cal", "silent_benign@cal",
                                               "open_f1@cal"]].mean().round(4)
               .sort_values("auroc", ascending=False))
    out = {"dataset_id": "dnn_edgeiiot", "condition": args.condition, "dropped_features": drop,
           "protocol": "LOFO; ekip 80/20 bölmesi (seed 42); eşik yalnızca bilinen kalibrasyon verisiyle",
           "settings": {k: v for k, v in vars(args).items() if k != "data_file"},
           "families": details, "results": rows,
           "mean_over_families": summary.reset_index().to_dict(orient="records"),
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out_dir, f"open_set_lofo_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot(rows, os.path.join(args.out_dir, f"fig_open_set_auroc_dnn_edgeiiot_{tag}.png"),
         f"Edge-IIoTset, {'+'.join(models)}, koşul {args.condition}: görülmemiş aile tespiti (AUROC)")
    print("\nAilelerin ortalaması:")
    print(summary.to_string())
    print(f"\nÇıktılar: {args.out_dir}  ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
