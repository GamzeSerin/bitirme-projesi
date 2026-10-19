"""Kişi 3 - Edge-IIoTset bölme sağlamlığı kontrolü: rastgele bölme vs zaman bazlı bölme.

Soru: Rastgele 80/20 bölmedeki yüksek skor, aynı bağlantının/oturumun paketlerinin hem
eğitimde hem testte bulunmasından mı geliyor? Model "gelecekteki" trafiğe genelleşiyor mu?

Yöntem (aynı satırlar, aynı sabit hiperparametreler):
  - Rastgele: ince etikete göre stratified 80/20, 5 farklı seed (bölme + model tohumu).
  - Zaman: her ince etiketin paketleri kayıt sırasına (CSV satır sırası) göre sıralanır; ilk %80 eğitim, son %20 test.
    Bölme sabittir; 5 seed yalnızca modelin rastgeleliğini değiştirir.
  - Her ikisi iki özellik setiyle: A (34 özellik) ve E (kısayolsuz: port, tcp.seq/ack,
    sayaç, zaman damgası çıkarılmış; bkz. ablation deneyi).
`frame.time` yalnızca sıralama için kullanılır, hiçbir zaman özellik değildir.

Temizlik asıl deneyle aynıdır (kimlik/metin/sabit sütunlar atılır, tekrarlar silinir,
çakışan satırlar atılır, etiket başına 40.000 üst sınır). Tekrarlardan kayıtta ilk görünen tutulur.

Çıktı: outputs/dnn_edgeiiot/split_check/ (split_check.json + grafik). Sözleşmeye dokunmaz.

Kullanım (Kaggle CPU, ~10-15 dk):
  python src/split_check_dnn_edgeiiot.py --data-file "<.../DNN-EdgeIIoT-dataset.csv>"
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ablation_dnn_edgeiiot import PORTS, SEQ_ACK, SUSPECT_IDS, build_models
from common.taxonomy import COMMON_CLASSES, edgeiiot_category
from model_dnn_edgeiiot import LABEL, misaligned_mask, select_features

OUT_DIR = "outputs/dnn_edgeiiot/split_check"
TIME_COL = "frame.time"
FEATURE_SETS = {"A_tam": [], "E_kisayolsuz": PORTS + SEQ_ACK + SUSPECT_IDS}


# ---------------------------------------------------------------- zaman
def parse_time(raw):
    """frame.time -> saniye (float). Önce tam tarih denenir; olmazsa saat:dakika:saniye.

    Dönüş: (saniye dizisi, yöntem adı, başarı oranı).
    """
    s = raw.astype(str).str.strip()
    no_date = s.str.match(r"^\d{4}\s+\d{1,2}:\d{2}").mean() > 0.5
    if not no_date:
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        ok = float(dt.notna().mean())
        if ok > 0.95:
            return (dt - dt.min()).dt.total_seconds().to_numpy(), "tam tarih", ok
    # Bazı sürümlerde ay/gün düşmüş: "2021 11:44:10.081753000" -> yalnızca günün saati
    m = s.str.extract(r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)")
    secs = (m[0].astype(float) * 3600 + m[1].astype(float) * 60 + m[2].astype(float)).to_numpy()
    return secs, "günün saati (tarih yok)", float(np.isfinite(secs).mean())


def load(data_file, cap, seed):
    df = pd.read_csv(data_file, low_memory=False)
    bad = misaligned_mask(df)
    print(f"Hizasız satır atıldı: {int(bad.sum())} {df.loc[bad, LABEL].value_counts().to_dict()}")
    df = df[~bad].reset_index(drop=True)
    stats = {"rows_raw": int(len(df) + bad.sum()), "rows_misaligned_dropped": int(bad.sum()), "time_examples": df[TIME_COL].astype(str).head(5).tolist()}
    secs, method, ok = parse_time(df[TIME_COL])
    stats.update(time_parse_method=method, time_parse_ok=round(ok, 4))
    print(f"frame.time örnekleri: {stats['time_examples']}")
    print(f"Zaman ayrıştırma: {method}, başarı %{100 * ok:.2f}")
    bad = ~np.isfinite(secs)
    if bad.any():
        raw = df.loc[bad, TIME_COL].astype(str)
        stats["time_unparsed_examples"] = raw.value_counts().head(8).to_dict()
        stats["time_unparsed_by_label"] = df.loc[bad, LABEL].astype(str).value_counts().to_dict()
        print(f"Ayrıştırılamayan örnekler: {stats['time_unparsed_examples']}")
        print(f"Ayrıştırılamayanların etiketleri: {stats['time_unparsed_by_label']}")
        # Dosya kayıt sırasıyla yazılmışsa en yakın önceki/sonraki paketin zamanı kullanılır;
        # böylece bu satırlar topluca test kümesinin sonuna yığılmaz.
        secs = pd.Series(secs).ffill().bfill().to_numpy()
        stats["time_unparsed_filled_from_neighbours"] = True

    feats, _ = select_features(df)
    df = df[feats + [LABEL]].copy()
    df["_t"] = secs
    df["_row"] = np.arange(len(df))  # dosyadaki sıra: zaman yoksa ya da eşitse
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    # Tekrarlardan dosyada (kayıtta) ilk görüneni kalsın
    df = df.drop_duplicates(subset=feats + [LABEL], keep="first")
    conflict = df.duplicated(subset=feats, keep=False).to_numpy()
    df = df[~conflict]
    stats.update(rows_after_dedup_and_conflicts=int(len(df)))

    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.choice(ix, size=min(len(ix), cap), replace=False)
                           for ix in df.groupby(LABEL).indices.values()])
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    stats["rows_after_sampling"] = int(len(df))
    fine = df[LABEL].astype(str).str.strip().str.upper().to_numpy()
    category = np.array([edgeiiot_category(l) for l in fine])
    return df[feats].astype(np.float32), fine, category, df["_t"].to_numpy(), df["_row"].to_numpy(), stats


def temporal_split(fine, t, row, frac=0.8, order_by="row"):
    """Her ince etiket için kayıt sırasına göre ilk %80 eğitim, son %20 test.

    order_by="row" (varsayılan): CSV'deki satır sırası. Dosya kayıt sırasıyla yazılmış
    (çoğu etikette zaman~satır sırası Spearman rho = 1,0). frame.time'da tarih olmadığı için
    gece yarısını geçen kayıtlarda (Ransomware rho=0,11, Normal rho=0,83) saat sırası
    yanıltıcıdır; satır sırası bu durumda daha güvenilir. order_by="time": saat sırası.
    """
    tr, te, info = [], [], {}
    for lab in sorted(set(fine)):
        ix = np.where(fine == lab)[0]
        if order_by == "time":
            key = np.where(np.isfinite(t[ix]), t[ix], np.inf)
            order = ix[np.lexsort((row[ix], key))]
        else:
            order = ix[np.argsort(row[ix], kind="stable")]
        cut = int(round(frac * len(order)))
        tr.append(order[:cut]); te.append(order[cut:])
        tt = t[ix][np.isfinite(t[ix])]
        # dosya sırası zamanla uyumlu mu? (Spearman)
        rho = pd.Series(t[ix]).corr(pd.Series(row[ix]), method="spearman") if len(tt) > 2 else None
        info[lab] = {"n": int(len(ix)), "time_span_s": round(float(tt.max() - tt.min()), 1) if len(tt) else None,
                     "spearman_time_vs_file_order": None if rho is None or np.isnan(rho) else round(float(rho), 3)}
    return np.concatenate(tr), np.concatenate(te), info


# ---------------------------------------------------------------- değerlendirme
def evaluate(est, X, y, fine, idx_tr, idx_te, classes):
    est.fit(X.iloc[idx_tr], y[idx_tr])
    pred = est.predict(X.iloc[idx_te])
    yte, fte = y[idx_te], fine[idx_te]
    benign = classes.index("BENIGN")
    attack, alarm = yte != benign, pred != benign
    return {
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "per_class_f1": dict(zip(classes, map(float, f1_score(yte, pred, average=None,
                                                             labels=range(len(classes)))))),
        "fine_recall": {lab: float((pred[fte == lab] == yte[fte == lab]).mean()) for lab in sorted(set(fte))},
        # Güvenlik bakışı: gerçek saldırılardan kaçı normal sanıldı / normal trafiğin kaçı alarm
        "missed_attack_rate": float((~alarm & attack).sum() / attack.sum()),
        "false_alarm_rate": float((alarm & ~attack).sum() / (~attack).sum()),
    }


def summarize(runs):
    keys = ["macro_f1", "missed_attack_rate", "false_alarm_rate"]
    out = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
               "std": round(float(np.std([r[k] for r in runs], ddof=1)), 4)} for k in keys}
    out["per_class_f1_mean"] = {c: round(float(np.mean([r["per_class_f1"][c] for r in runs])), 4)
                                for c in runs[0]["per_class_f1"]}
    out["fine_recall_mean"] = {c: round(float(np.mean([r["fine_recall"][c] for r in runs])), 4)
                               for c in runs[0]["fine_recall"]}
    out["macro_f1_per_seed"] = [round(r["macro_f1"], 4) for r in runs]
    return out


def plot(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fsets, splits = list(FEATURE_SETS), ["rastgele", "zaman"]
    colors = {"rastgele": "#2a78d6", "zaman": "#d9822b"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.38
    for j, sp in enumerate(splits):
        xs = np.arange(len(fsets)) + (j - 0.5) * w
        m = [res[f][sp]["RF"]["macro_f1"]["mean"] for f in fsets]
        sd = [res[f][sp]["RF"]["macro_f1"]["std"] for f in fsets]
        ax.bar(xs, m, width=w * 0.92, yerr=sd, capsize=3, color=colors[sp], label=f"{sp} bölme")
        for x, v in zip(xs, m):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(fsets)), ["A: Tam (34)", "E: Kısayolsuz (28)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("RF test macro-F1 (5 seed ort. ± ss)")
    ax.set_title("Edge-IIoTset: rastgele vs zaman bazlı bölme", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "split_check_macro_f1_RF.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--models", default="RF,DT")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--order-by", choices=["row", "time"], default="row",
                   help="Zaman bölmesinde sıralama: CSV satır sırası (varsayılan) ya da frame.time saati")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.seeds = min(args.cap, 3000), "0,1"
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    X, fine, category, t, row, stats = load(args.data_file, args.cap, seed=42)
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    print(f"Örneklem: {len(y)} satır, {X.shape[1]} özellik")

    tr_time, te_time, time_info = temporal_split(fine, t, row, order_by=args.order_by)
    for lab, v in time_info.items():
        print(f"  {lab:22s} n={v['n']:6d}  süre={v['time_span_s']} s  "
              f"zaman~dosya sırası ρ={v['spearman_time_vs_file_order']}")

    res = {}
    for fs, drop in FEATURE_SETS.items():
        Xf = X.drop(columns=[c for c in drop if c in X.columns])
        res[fs] = {"rastgele": {}, "zaman": {}}
        for m in [m for m in args.models.split(",")]:
            runs = {"rastgele": [], "zaman": []}
            for s in seeds:
                est = clone(build_models(s, args.n_jobs)[m])
                tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=fine, random_state=s)
                t0 = time.perf_counter()
                runs["rastgele"].append(evaluate(clone(est), Xf, y, fine, tr, te, classes))
                runs["zaman"].append(evaluate(clone(est), Xf, y, fine, tr_time, te_time, classes))
                print(f"{fs} {m} seed={s}: rastgele {runs['rastgele'][-1]['macro_f1']:.4f}  "
                      f"zaman {runs['zaman'][-1]['macro_f1']:.4f}  ({time.perf_counter() - t0:.0f} s)")
            for sp in runs:
                res[fs][sp][m] = summarize(runs[sp])

    out = {"dataset_id": "dnn_edgeiiot", "dataset_name": "Edge-IIoTset",
           "protocol": "Aynı satırlar ve sabit hiperparametreler. Rastgele: ince etikete göre "
                       "stratified 80/20, her seed farklı bölme. Zaman: her ince etiketin zamanca ilk "
                       "%80'i eğitim, son %20'si test (sabit bölme). frame.time özellik olarak kullanılmaz.",
           "seeds": seeds, "temporal_order_by": args.order_by, "cap_per_fine_label": args.cap, "quick": bool(args.quick),
           "feature_sets": FEATURE_SETS, "classes": classes,
           "data_prep": stats, "time_split_by_label": time_info, "results": res}
    with open(os.path.join(args.out_dir, "split_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if "RF" in args.models.split(","):
        plot(res, args.out_dir)

    print("\nÖzet (test macro-F1, ort. ± ss | kaçırılan saldırı | yanlış alarm):")
    for fs in res:
        for sp in res[fs]:
            for m, r in res[fs][sp].items():
                print(f"  {fs:13s} {sp:9s} {m}: {r['macro_f1']['mean']:.4f} ± {r['macro_f1']['std']:.4f}"
                      f" | {r['missed_attack_rate']['mean']:.4f} | {r['false_alarm_rate']['mean']:.4f}")
    print(f"Çıktılar: {args.out_dir}")


if __name__ == "__main__":
    main()
