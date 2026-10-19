"""Kişi 3 - dnn_edgeiiot (Edge-IIoTset) kapalı-küme deneyi.

Eğitim/OOF/test/SHAP akışı Kişi 1'in src/model_ml.py dosyasındakiyle birebir aynıdır
(oradaki fonksiyonlar yeniden kullanılır); yalnızca veri okuma ve temizlik bu veri
setine özgüdür. Böylece üç ortamın sonuçları aynı protokolle karşılaştırılabilir.

Akış:
  0) DNN-EdgeIIoT-dataset.csv oku -> kimlik/zaman/ham içerik (metin) ve sabit sütunları at
     -> kimlik gibi davranan sayısal sütunları at (tcp.ack_raw, checksum'lar, udp.stream)
     -> inf/NaN temizle -> tekrarları sil -> (opsiyonel) çakışan satırları at
     -> ince etiket başına üst sınırla örnekle -> 15 etiketi ortak kategorilere eşle
  1-6) model_ml.py ile aynı: stratified 80/20 (seed=42), OOF, test, SHAP, sözleşme çıktısı
  Modeller: DT, RF, XGB, LGBM, LR (model_ml.build_models) + TabM (src/tabm_classifier.py;
  hiperparametre araması yok, resmi varsayılanlar ve early stopping; GPU önerilir)

Kullanım (Kaggle):
  python src/model_dnn_edgeiiot.py \
      --data-file "/kaggle/input/.../Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv" \
      --work-dir /kaggle/working/work/dnn_edgeiiot --drop-conflicts --shap
  python src/model_dnn_edgeiiot.py --data-file <csv> --quick --out-dir outputs_quick/dnn_edgeiiot
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import model_ml as shared  # Kişi 1'in eğitim/OOF/SHAP fonksiyonları (ortak protokol)
from common import contract
from common.taxonomy import COMMON_CLASSES, edgeiiot_category

DATASET_ID = "dnn_edgeiiot"
LABEL = "Attack_type"
LABEL_COLS = {"Attack_type", "Attack_label"}
# Aşama 0 denetimi: sayısal görünen ama kayda/bağlantıya özgü kimlik gibi davranan sütunlar.
# Bunlar atılmadığında 197 bin satırlık etiket çakışması "gizleniyor" (model kimliği ezberliyor).
ID_LIKE_NUMERIC = ["tcp.ack_raw", "tcp.checksum", "icmp.checksum", "udp.stream"]
# 19.10.2026 denetimi: DDoS_UDP ve MITM satırlarında sütunlar kaymış (frame.time'da IP adresi
# ya da sayı var; MITM'de ARP alanları sıfır, udp.port'ta mDNS metni). Kayma yayımlanan
# DNN, ML ve ham "Attack traffic" CSV'lerinin hepsinde aynı; güvenilir bir hizalama yok.
# Bu satırlar frame.time biçimine bakılarak bulunur ve atılır.
TIME_COL = "frame.time"
TIME_PATTERN = r"^\s*\d{4}\s+\d{1,2}:\d{2}:\d{2}"


def misaligned_mask(df):
    """frame.time bir zaman damgası değilse satır hizasızdır (bkz. README, bulgu 7)."""
    return ~df[TIME_COL].astype(str).str.match(TIME_PATTERN).to_numpy()


# ---------------------------------------------------------------- veri
def select_features(df, drop_features=()):
    """Sayısal, sabit olmayan, kimlik gibi davranmayan özellikler + çıkarılanların gerekçesi."""
    cand = [c for c in df.columns if c not in LABEL_COLS]
    text = [c for c in cand if not pd.api.types.is_numeric_dtype(df[c])]
    constant = [c for c in cand if c not in text and df[c].nunique(dropna=False) <= 1]
    id_like = [c for c in ID_LIKE_NUMERIC if c in df.columns]
    extra = [c for c in drop_features if c in df.columns]
    dropped = set(text) | set(constant) | set(id_like) | set(extra)
    feats = [c for c in cand if c not in dropped]
    return feats, {"text_or_identifier": text, "constant": constant,
                   "id_like_numeric": id_like, "user_dropped": extra}


def load_data(data_file, cap, seed, drop_features=(), drop_conflicts=False, keep_misaligned=False):
    print(f"Okunuyor: {data_file}")
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file, low_memory=False)
    stats = {"file": os.path.basename(data_file), "rows_raw": int(len(df))}

    bad = misaligned_mask(df)
    stats["rows_misaligned"] = int(bad.sum())
    stats["misaligned_by_label"] = {str(k): int(v) for k, v in
                                    df.loc[bad, LABEL].value_counts().items()}
    stats["misaligned_dropped"] = not keep_misaligned
    print(f"Hizasız satır (frame.time zaman değil): {stats['rows_misaligned']} "
          f"{stats['misaligned_by_label']} -> {'tutuldu' if keep_misaligned else 'atıldı'}")
    if not keep_misaligned:
        df = df[~bad]

    feats, dropped = select_features(df, drop_features)
    stats["dropped_columns"] = dropped
    df = df[feats + [LABEL]].copy()
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    stats["rows_after_nan_inf"] = int(len(df))

    df = df.drop_duplicates()
    stats["rows_after_dedup"] = int(len(df))
    # Aynı özellik vektörü, farklı etiket: paket düzeyinde etiketler kayıt dosyası bazında
    # verildiği için saldırı kayıtlarına karışan sıradan paketler (ör. ARP) saldırı etiketi almış.
    conflict = df.duplicated(subset=feats, keep=False).to_numpy()
    stats["rows_conflicting_labels"] = int(conflict.sum())
    stats["conflicting_by_label"] = {str(k): int(v) for k, v in
                                     df.loc[conflict, LABEL].value_counts().items()}
    if drop_conflicts:
        df = df[~conflict]
    stats["conflicts_dropped"] = bool(drop_conflicts)

    rng = np.random.default_rng(seed)
    groups = df.groupby(LABEL).indices
    keep = np.concatenate([rng.choice(ix, size=min(len(ix), cap), replace=False)
                           for ix in groups.values()])
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    stats["cap_per_fine_label"] = cap
    stats["rows_after_sampling"] = int(len(df))
    stats["fine_label_counts"] = {str(k): int(v) for k, v in df[LABEL].value_counts().items()}

    fine = df[LABEL].astype(str).str.strip().str.upper().to_numpy()
    category = np.array([edgeiiot_category(l) for l in fine])
    X = df.drop(columns=[LABEL]).astype(np.float32)
    stats["category_counts"] = {str(k): int(v) for k, v in pd.Series(category).value_counts().items()}
    return X, fine, category, stats


# ---------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True,
                   help="DNN-EdgeIIoT-dataset.csv (veya Aşama 0'ın .parquet çıktısı)")
    p.add_argument("--out-dir", default=f"outputs/{DATASET_ID}")
    p.add_argument("--work-dir", default=f"work/{DATASET_ID}")
    p.add_argument("--cap", type=int, default=40000, help="İnce etiket başına en fazla satır")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="DT,RF,XGB,LGBM,LR,TabM")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-iter", type=int, default=30)  # ekip protokolü (Kişi 1, 19.10.2026)
    p.add_argument("--tune-rows", type=int, default=200000)
    p.add_argument("--shap", action="store_true")
    p.add_argument("--shap-per-class", type=int, default=100)
    p.add_argument("--pred-per-class", type=int, default=0,
                   help="0 = tüm test seti (ekip protokolü; DSS modellerle aynı satırlarda değerlendirilir)")
    p.add_argument("--drop-features", default="",
                   help="Atılacak ek özellikler, ör. 'tcp.dstport,udp.port,tcp.seq,tcp.ack' (ablation)")
    p.add_argument("--drop-conflicts", action="store_true",
                   help="Aynı özellik + farklı etiketli satırları at (dnn_edgeiiot için önerilir)")
    p.add_argument("--keep-misaligned", action="store_true",
                   help="Sütunları kaymış DDoS_UDP/MITM satırlarını tut (yalnızca duyarlılık analizi)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.folds = min(args.cap, 3000), 3

    drop = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    X, fine, category, data_stats = load_data(args.data_file, args.cap, args.seed,
                                              drop, args.drop_conflicts, args.keep_misaligned)
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    print(f"Örneklem: {len(y)} satır, {X.shape[1]} özellik, {len(classes)} kategori")

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=fine, random_state=args.seed)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    cfg = {k: getattr(args, k) for k in ("data_file", "cap", "folds", "models", "tune", "n_iter",
                                         "tune_rows", "drop_features", "drop_conflicts", "keep_misaligned", "seed",
                                         "quick")}
    cfg["data_file"] = os.path.basename(cfg["data_file"])
    run_dir = os.path.join(args.work_dir, hashlib.md5(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8])
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    all_models = shared.build_models(args.seed, args.n_jobs, args.gpu, args.quick)
    if "TabM" in args.models.split(","):
        try:
            import rtdl_num_embeddings, tabm, torch  # noqa: F401
            from tabm_classifier import TabMClassifier
            all_models["TabM"] = TabMClassifier(max_epochs=3 if args.quick else 60,
                                                random_state=args.seed)
            print(f"TabM: torch {torch.__version__}, cihaz "
                  f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
        except ImportError as e:
            print(f"! TabM atlanıyor ({e}); kurulum: pip install tabm rtdl_num_embeddings")
    names = [m for m in args.models.split(",") if m in all_models]
    results = {}
    for name in names:
        cache = os.path.join(run_dir, f"result_{name}.joblib")
        if os.path.exists(cache) and not args.force:
            print(f"{name}: kayıtlı sonuç kullanılıyor ({cache})")
            results[name] = joblib.load(cache)
            continue
        print(f"{name} eğitiliyor...")
        # TabM için hiperparametre araması yok (resmi varsayılanlar + early stopping)
        run_args = argparse.Namespace(**{**vars(args), "tune": args.tune and name in shared.PARAM_SPACE})
        res = shared.run_model(name, all_models[name], Xtr, ytr, fine[idx_tr], Xte,
                               len(classes), run_args)
        joblib.dump(res.pop("model"), os.path.join(run_dir, "models", f"{name}.joblib"))
        joblib.dump(res, cache)
        results[name] = res

    metrics, oof_metrics = {}, {}
    for name, r in results.items():
        oof_metrics[name] = contract.evaluate(ytr, r["oof_proba"], classes)
        metrics[name] = {
            "test": contract.evaluate(yte, r["test_proba"], classes),
            "oof": oof_metrics[name],
            "test_fine_label_recall": shared.fine_label_recall(yte, r["test_proba"], fine[idx_te]),
            "timing": {"fit_seconds": r["fit_seconds"],
                       "predict_ms_per_1k": r["predict_ms_per_1k"]},
            "params": r["params"],
        }
        print(f"{name:5s} OOF macro-F1={oof_metrics[name]['macro_f1']:.4f}  "
              f"test macro-F1={metrics[name]['test']['macro_f1']:.4f}")

    primary = max(oof_metrics, key=lambda m: oof_metrics[m]["macro_f1"])
    best_per_class = {c: max(oof_metrics, key=lambda m: oof_metrics[m]["per_class"][c]["f1"])
                      for c in classes}

    sample = (contract.stratified_sample(yte, args.pred_per_class, args.seed)
              if args.pred_per_class > 0 else np.arange(len(yte)))
    preds = pd.DataFrame({"y_true": [classes[i] for i in yte[sample]],
                          "fine_label": fine[idx_te][sample]})
    for name, r in results.items():
        for i, c in enumerate(classes):
            preds[contract.proba_column(name, c)] = r["test_proba"][sample, i]

    shap_top = None
    if args.shap:
        shap_model = primary if primary in shared.TREE_MODELS else next(
            (m for m in names if m in shared.TREE_MODELS), None)
        if shap_model:
            print(f"SHAP hesaplanıyor ({shap_model})...")
            model = joblib.load(os.path.join(run_dir, "models", f"{shap_model}.joblib"))
            shap_top = shared.shap_top_features(model, shap_model, Xte, yte, classes,
                                                args.shap_per_class, args.seed,
                                                fig_dir=os.path.join(args.out_dir, "figures"))

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": "Edge-IIoTset (DNN-EdgeIIoT-dataset.csv)",
        "environment": "Endüstriyel / edge IoT test ortamı (7 katmanlı, 10+ sensör türü, Modbus, MQTT)",
        "owner": "Kişi 3",
        "label_level": "category",
        "classes": classes,
        "features": list(X.columns),
        "models": list(results),
        "primary_model": primary,
        "best_model_per_class": best_per_class,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "cv_folds": args.folds,
        "tuned": bool(args.tune),
        "seed": args.seed,
        "runtime": {"gpu_used_for_xgb": bool(args.gpu), "gpu": shared.gpu_name()},
        "data_prep": data_stats,
    }
    contract.write_outputs(args.out_dir, manifest, metrics,
                           contract.reliability_from_oof(oof_metrics), preds, shap_top)
    print(f"\nBirincil model: {primary}. Çıktılar: {args.out_dir}")
    for c, m in best_per_class.items():
        print(f"  {c:11s} {m}  F1={oof_metrics[m]['per_class'][c]['f1']:.4f}")


if __name__ == "__main__":
    main()
