"""Görülmemiş saldırı deneyi (LOFO): ekibin ORTAK aracı, veri setinden bağımsız.

Veri setine özel her şey (yükleyici, aileler, koşullar) common/open_set_registry.py içindedir:
  python src/open_set_lofo.py --dataset dnn_edgeiiot --data-file <DNN-EdgeIIoT-dataset.csv> ...
  python src/open_set_lofo.py --dataset dataset_23   --data-file <Merged*.csv klasörü> ...
Bir aile birden fazla kategori kapsayabilir (ör. CICIoT2023'te FLOOD = DOS + DDOS).
Eski komut (open_set_lofo_dnn_edgeiiot.py) aynen çalışır; --dataset dnn_edgeiiot ile bu betiği çağırır.

Soru: Bir saldırı ailesi eğitimde HİÇ görülmediyse, model onu "bilinmeyen" olarak işaretleyebilir mi,
yoksa sessizce bilinen bir sınıfa (en kötüsü BENIGN) mı atar?

Protokol (leave-one-family-out, LOFO):
  - Ekip bölmesi: tüm temiz veri 80/20, ince etikete göre katmanlı, seed 42 (asıl deneyle aynı test).
  - Her turda bir saldırı ailesi U (kayıttaki aileler; Edge-IIoTset: DDOS, RECON, WEB, BRUTEFORCE,
    MALWARE) eğitimden tamamen çıkarılır; ailenin kapsadığı tüm kategoriler "bilinmeyen" olur.
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

Kısayol alanlar bilinmeyen tespitini de etkileyebileceği için deney kayıttaki koşullarda
çalıştırılır (A tam, C ana koşul, E en sert); koşul adları veri setine göre değişir.

Çalıştırma:
  python src/open_set_lofo.py --dataset dnn_edgeiiot --data-file <csv> --models RF,TabM --condition C_seq_ack_yok
  python src/open_set_lofo.py --dataset dataset_23 --data-file <klasör> --models RF,TabM --condition C_ttl_yok
  ... --quick                                       # tesisat denemesi (~1 dk)
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

from ablation_dnn_edgeiiot import build_models
from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES

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
    if name in ("TabM", "FPTabM"):
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
def plot(rows, out_path, title, families=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    df["yöntem"] = df["model"] + " · " + df["score"]
    piv = df.pivot_table(index="yöntem", columns="family", values="auroc")
    order = families or FAMILIES
    piv = piv[[f for f in order if f in piv.columns] + [f for f in piv.columns if f not in order]]
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
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True, help="Dosya ya da klasör (veri setinin yükleyicisine göre)")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/open_set")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42, help="Veri bölmesi ve örneklem (ekip protokolü: 42)")
    p.add_argument("--model-seed", type=int, default=None,
                   help="Yalnızca modelin eğitim rastgeleliği (tohum kontrolü); varsayılan --seed")
    p.add_argument("--cp-config", default="",
                   help="cp_tune JSON'u: CP-TabM her aile için iç provanın seçtiği kör oranıyla eğitilir")
    p.add_argument("--condition", default="A_tam", help="Kayıttaki koşul adı (veri setine göre)")
    p.add_argument("--models", default="RF",
                   help="Virgülle: RF, TabM, FPTabM (özellik-bölümlü), CPTabM (önerilen), "
                        "CPTabM-noE (enerji kaybı kapalı bileşen testi)")
    p.add_argument("--families", default="", help="Virgülle; boş = kayıttaki tüm aileler")
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95, help="Kabul edilecek bilinen oranı")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--cp-blind-frac", type=float, default=0.25, help="CP-TabM: sınıf başına kör üye oranı")
    p.add_argument("--cp-lam", type=float, default=0.1, help="CP-TabM: enerji sınırı kaybının ağırlığı")
    p.add_argument("--cp-m-in", type=float, default=-6.0)
    p.add_argument("--cp-m-out", type=float, default=-1.0)
    p.add_argument("--cp-feature-frac", type=float, default=None,
                   help="CP-TabM üyelerine ayrıca özellik alt kümesi (None: kapalı)")
    p.add_argument("--fp-frac", type=float, default=0.5, help="FPTabM: her üyenin gördüğü özellik oranı")
    p.add_argument("--label-level", default="category", choices=["category", "fine"],
                   help="fine: alt türlerle eğit, tahmini kategoriye topla")
    p.add_argument("--tag-suffix", default="", help="Çıktı adına ek (ör. _fp05)")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap = 3000
    spec = registry.get(args.dataset)
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}/open_set"
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # Satırlar tam koşulda seçilir, sütunlar sonra çıkarılır: her koşulda eğitim/test satırları aynıdır.
    X, fine, category, data_stats = spec["loader"](args.data_file, args.cap, args.seed)
    X, drop = registry.apply_condition(X, spec, args.condition)
    benign = spec["benign"]
    present = [c for c in COMMON_CLASSES if c in set(category)] + sorted(set(category) - set(COMMON_CLASSES))
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, {X.shape[1]} özellik, kategoriler {present} "
          f"(koşul {args.condition}, çıkarılan {drop})")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    fam_def = spec["families"]
    wanted = [f for f in args.families.split(",") if f] or list(fam_def)
    unknown_names = [f for f in wanted if f not in fam_def]
    if unknown_names:
        raise SystemExit(f"Kayıtta olmayan aile: {unknown_names} (kayıtlı: {list(fam_def)})")
    families = [f for f in wanted if set(fam_def[f]) & set(present)]
    skipped = [f for f in wanted if f not in families]
    if skipped:
        print(f"! Veride hiç satırı olmayan aileler atlandı: {skipped}")

    rows, details = [], {}
    mseed = args.seed if args.model_seed is None else args.model_seed
    cp_cfg = {}
    if args.cp_config:
        with open(args.cp_config, encoding="utf-8") as f:
            cp_cfg = {fam: v["chosen"] for fam, v in json.load(f)["families"].items()}
        print(f"CP-TabM ayarları iç provadan: { {k: v['blind_frac'] for k, v in cp_cfg.items()} }")
    for fam in families:
        t0 = time.perf_counter()
        fam_cats = list(fam_def[fam])
        known = [c for c in present if c not in fam_cats]
        tr = idx_tr[~np.isin(category[idx_tr], fam_cats)]
        tr_fit, tr_cal = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr],
                                          random_state=args.seed)
        Xfit, Xcal, Xte = X.iloc[tr_fit], X.iloc[tr_cal], X.iloc[idx_te]
        if args.label_level == "fine":
            # Alt türlerle eğitim: model bilinen ailelerin ince etiketlerini öğrenir; tahmin kategoriye toplanır
            labels = sorted(set(fine[tr_fit]))
            yfit = np.array([labels.index(c) for c in fine[tr_fit]])
            label_cat = np.array([spec["category_of"](c) for c in labels])
        else:
            labels = known
            yfit = np.array([known.index(c) for c in category[tr_fit]])
            label_cat = np.array(known)
        is_unk = np.isin(category[idx_te], fam_cats)
        y_te_names = category[idx_te]
        print(f"\n=== Bilinmeyen aile: {fam} {fam_cats}  (eğitim {len(tr_fit)}, kalibrasyon {len(tr_cal)}, "
              f"test bilinen {int((~is_unk).sum())}, bilinmeyen {int(is_unk.sum())}) ===")
        details[fam] = {"categories": fam_cats, "known_classes": known, "n_fit": int(len(tr_fit)), "n_cal": int(len(tr_cal)),
                        "n_test_known": int((~is_unk).sum()), "n_test_unknown": int(is_unk.sum()),
                        "unknown_fine_labels": sorted(set(fine[idx_te][is_unk])), "models": {}}

        knn = KNNScore(seed=args.seed).fit(Xfit)
        knn_cal, knn_te = knn.score(Xcal), knn.score(Xte)

        for name in models:
            if name in ("TabM", "FPTabM"):
                from tabm_classifier import TabMClassifier
                model = TabMClassifier(max_epochs=3 if args.quick else 60, random_state=mseed,
                                       verbose=False, feature_frac=args.fp_frac if name == "FPTabM" else None)
            elif name.startswith("CPTabM"):
                from cp_tabm_classifier import CPTabMClassifier
                model = CPTabMClassifier(blind_frac=cp_cfg.get(fam, {}).get("blind_frac", args.cp_blind_frac),
                                         lam_energy=0.0 if name == "CPTabM-noE" else args.cp_lam,
                                         m_in=args.cp_m_in, m_out=args.cp_m_out,
                                         max_epochs=3 if args.quick else 60, random_state=mseed,
                                         verbose=False, feature_frac=args.cp_feature_frac)
            else:
                model = clone(build_models(mseed, args.n_jobs)[name])
            tm = time.perf_counter()
            model.fit(Xfit, yfit)
            fit_s = time.perf_counter() - tm
            P_cal, S_cal = model_scores(name, model, Xcal)
            P_te, S_te = model_scores(name, model, Xte)
            pred = label_cat[P_te.argmax(1)]
            closed_f1 = f1_score(y_te_names[~is_unk], pred[~is_unk], average="macro")
            assigned = pd.Series(pred[is_unk]).value_counts(normalize=True).round(4).to_dict()
            details[fam]["models"][name] = {"closed_f1_known": round(float(closed_f1), 4),
                                            "fit_seconds": round(fit_s, 1),
                                            "unknown_assigned_to": assigned}
            details[fam]["models"][name]["model_seed"] = mseed
            if name.startswith("CPTabM") and fam in cp_cfg:
                details[fam]["models"][name]["chosen_by_inner"] = cp_cfg[fam]
            if name in ("TabM", "FPTabM") or name.startswith("CPTabM"):
                details[fam]["models"][name]["best_epoch"] = int(model.best_epoch_)
            if name.startswith("CPTabM"):
                info = details[fam]["models"][name]
                info["blind_matrix"] = model.A_.astype(int).tolist()
                info["energy_accept_median"] = round(float(np.median(model.e_acc_)), 3)
                info["energy_reject_median"] = round(float(np.median(model.e_rej_)), 3)
                info["energy_gap_median"] = round(model.energy_gap_, 3)
                # Açıklama örnekleri: görülmemiş aileden 5 satır (XAI bölümü için)
                ex_idx = np.flatnonzero(is_unk)[:5]
                info["explain_unknown_examples"] = model.explain(Xte.iloc[ex_idx], labels)
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
                             pred, y_te_names, benign, args.target)
                row = {"family": fam, "model": "bağımsız" if sname == "knn" else name, "score": sname, **m}
                rows.append(row)
                print(f"    {row['model']:8s} {sname:9s} AUROC {m['auroc']:.3f}  FPR95 {m['fpr95']:.3f}  "
                      f"yakalanan {m['detect@cal']:.3f}  BENIGN'e kaçan {m['silent_benign@cal']:.3f}  "
                      f"open-F1 {m['open_f1@cal']:.3f}")
        print(f"  ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    tag = f"{'_'.join(models)}_{args.condition}" + ("_fine" if args.label_level == "fine" else "") + args.tag_suffix
    df.to_csv(os.path.join(args.out_dir, f"open_set_lofo_{tag}.csv"), index=False)
    summary = (df.groupby(["model", "score"])[["auroc", "fpr95", "detect@cal", "silent_benign@cal",
                                               "open_f1@cal"]].mean().round(4)
               .sort_values("auroc", ascending=False))
    out = {"dataset_id": args.dataset, "dataset_name": spec["name"], "condition": args.condition,
           "dropped_features": drop, "family_definition": {f: fam_def[f] for f in families},
           "data_preparation": data_stats,
           "protocol": "LOFO; ekip 80/20 bölmesi (seed 42); eşik yalnızca bilinen kalibrasyon verisiyle",
           "settings": {k: v for k, v in vars(args).items() if k != "data_file"},
           "families": details, "results": rows,
           "mean_over_families": summary.reset_index().to_dict(orient="records"),
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out_dir, f"open_set_lofo_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot(rows, os.path.join(args.out_dir, f"fig_open_set_auroc_{args.dataset}_{tag}.png"),
         f"{spec['name']}, {'+'.join(models)}, koşul {args.condition}: görülmemiş aile tespiti (AUROC)",
         families)
    print("\nAilelerin ortalaması:")
    print(summary.to_string())
    print(f"\nÇıktılar: {args.out_dir}  ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
