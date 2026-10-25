"""Kişi 1 - dataset_23 (CICIoT2023) klasik ML deneyi.

Akış (metodolojiyle birebir):
  0) Merged*.csv dosyalarını birleştir -> inf/NaN temizle -> tekrarları sil
     -> gereksiz sütunları at -> ince etiket başına üst sınır ile örnekle
     -> 34 etiketi 8 ortak kategoriye eşle
  1) İnce etikete göre stratified 80/20 split, test seti kilitli
  2) (opsiyonel --tune) eğitim alt örnekleminde RandomizedSearch
  3) Eğitim setinde stratified k-fold -> OOF olasılıkları (DSS ağırlıkları buradan)
  4) Tüm eğitim setinde yeniden eğit -> test setinde değerlendir
  5) (opsiyonel --shap) birincil model için sınıf bazlı SHAP
  6) outputs/dataset_23/ altına sözleşme formatında yaz

Kullanım:
  python src/model_ml.py --data-dir /content/drive/MyDrive/iot-data/dataset_23 \
      --work-dir /content/drive/MyDrive/iot-work/dataset_23 --shap
  python src/model_ml.py --data-dir data/dataset_23 --quick      # birkaç dakikalık deneme
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from common import contract
from common.taxonomy import COMMON_CLASSES, ciciot2023_category

DATASET_ID = "dataset_23"
LABEL = "Label"
# Veride doğrulandı: Tot size == AVG (satırların %100'ü), Variance == Std^2 (~%100)
REDUNDANT = ["Tot size", "Variance"]
TREE_MODELS = {"DT", "RF", "XGB", "LGBM"}
NEEDS_SAMPLE_WEIGHT = set()  # XGB dengelemesini EarlyStopping sarmalayıcısı yapıyor


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def gpu_name():
    """nvidia-smi varsa ekran kartının adı (makaledeki deney ortamı için)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


CACHE_MODELS_KEY = "DT,RF,XGB,LGBM,LR"  # önbellek klasörü için sabit (bkz. main)


def model_key(name, model):
    """Model tanımı + arama uzayının kısa parmak izi (bellek adresleri temizlenir)."""
    import re
    text = re.sub(r" at 0x[0-9A-Fa-f]+", "", repr(model)) + repr(PARAM_SPACE.get(name))
    if hasattr(model, "VERSION"):           # yalnızca sürümlü sınıflarda (EarlyStopping)
        text += f"v{model.VERSION}"
    return hashlib.md5(text.encode()).hexdigest()[:6]


class EarlyStopping(ClassifierMixin, BaseEstimator):
    """Boosting modelleri için early stopping sarmalayıcısı.

    Ağaç sayısı tuning'de sabitlenmez: her eğitimde, VERİLEN eğitim verisinin %10'u
    (stratified) doğrulama olarak ayrılır ve doğrulama kaybı `rounds` tur boyunca
    iyileşmeyince eğitim durur. Alt örneklemde seçilen az ağacın tam veride yetersiz
    kalması (underfitting) böylece önlenir. Test seti hiç kullanılmaz.

    Durma kriteri AĞIRLIKSIZ doğrulama kaybıdır. Eğitimde sınıf ağırlıkları kullanılır,
    ama ağırlıklı doğrulama kaybı nadir sınıfların birkaç düzine örneği ezberlenince
    erken yükselir (teşhis: ağırlıklı kayıp 64. turda, macro-F1 ve ağırlıksız kayıp
    ~400. turda en iyi). Bu yüzden durma kararı ağırlıksız kayba göre verilir.
    """
    VERSION = 2  # davranış değişince önbelleği geçersiz kılar (model_key'e girer)

    def __init__(self, estimator=None, val_size=0.1, rounds=50, balanced=False, random_state=42):
        self.estimator = estimator
        self.val_size = val_size
        self.rounds = rounds
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=self.val_size, stratify=y,
                                          random_state=self.random_state)
        est = clone(self.estimator)
        kind = type(est).__name__
        if kind == "XGBClassifier":
            est.set_params(early_stopping_rounds=self.rounds)
            est.fit(Xt, yt, sample_weight=compute_sample_weight("balanced", yt) if self.balanced else None,
                    eval_set=[(Xv, yv)], verbose=False)
            self.best_iteration_ = int(est.best_iteration) + 1
        elif kind == "LGBMClassifier":
            import lightgbm as lgb
            est.fit(Xt, yt, eval_set=[(Xv, yv)],
                    callbacks=[lgb.early_stopping(self.rounds, verbose=False)])
            self.best_iteration_ = int(est.best_iteration_)
        else:
            raise ValueError(f"EarlyStopping desteklemiyor: {kind}")
        self.estimator_ = est
        self.classes_ = est.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def tabm_model(seed, quick):
    """TabM (Gorishniy vd., ICLR 2025): resmi varsayılanlar + early stopping, arama YOK.

    Kurulu değilse None döner ve açık bir uyarı basılır (diğer modeller etkilenmez).
    Kurulum: pip install tabm rtdl_num_embeddings (+ GPU için CUDA'lı torch).
    """
    try:
        import rtdl_num_embeddings, tabm, torch  # noqa: F401
        from tabm_classifier import TabMClassifier
    except ImportError as e:
        print(f"! TabM atlanıyor ({e}); kurulum: pip install tabm rtdl_num_embeddings")
        return None
    print(f"TabM: torch {torch.__version__}, cihaz "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    return TabMClassifier(max_epochs=3 if quick else 60, random_state=seed)


def build_models(seed, n_jobs, gpu, quick, tabm=False):
    # Boosting: üst sınır geniş, gerçek ağaç sayısını early stopping belirler
    n_est = 300 if quick else 3000
    models = {
        "DT": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=50 if quick else 200,
                                     class_weight="balanced_subsample",
                                     n_jobs=n_jobs, random_state=seed),
        "LR": Pipeline([
            ("log", FunctionTransformer(signed_log1p)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),  # 1000 adım yakınsamıyordu
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = EarlyStopping(XGBClassifier(
            n_estimators=n_est, max_depth=8, learning_rate=0.1, tree_method="hist",
            device="cuda" if gpu else "cpu", n_jobs=n_jobs, random_state=seed,
            eval_metric="mlogloss"), balanced=True, random_state=seed)
    except ImportError:
        print("! xgboost yok, XGB atlanıyor")
    try:
        from lightgbm import LGBMClassifier
        models["LGBM"] = EarlyStopping(LGBMClassifier(
            n_estimators=n_est, learning_rate=0.1, num_leaves=63,
            class_weight="balanced", n_jobs=n_jobs, random_state=seed, verbose=-1),
            random_state=seed)
    except ImportError:
        print("! lightgbm yok, LGBM atlanıyor")
    if tabm:  # yalnızca istenirse: torch'u gereksiz yere yüklemesin
        model = tabm_model(seed, quick)
        if model is not None:
            models["TabM"] = model
    return models


PARAM_SPACE = {
    "DT": {"max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 5, 10],
           "criterion": ["gini", "entropy"]},
    "RF": {"n_estimators": [100, 200, 400], "max_depth": [None, 20, 40],
           "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5]},
    # Boosting'te n_estimators aranmaz; early stopping belirler (EarlyStopping sınıfı)
    "XGB": {"estimator__max_depth": [4, 6, 8, 10], "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__subsample": [0.8, 1.0], "estimator__colsample_bytree": [0.8, 1.0],
            "estimator__min_child_weight": [1, 5]},
    "LGBM": {"estimator__num_leaves": [31, 63, 127, 255],
             "estimator__learning_rate": [0.05, 0.1, 0.2],
             "estimator__min_child_samples": [10, 20, 50],
             "estimator__colsample_bytree": [0.8, 1.0]},
    "LR": {"clf__C": [0.1, 1.0, 10.0]},
}


# ---------------------------------------------------------------- veri
def load_data(data_dir, cap, seed, drop_features=(), drop_conflicts=False):
    files = sorted(glob.glob(os.path.join(data_dir, "Merged*.csv")))
    if not files:
        raise FileNotFoundError(f"{data_dir} içinde Merged*.csv bulunamadı")
    print(f"{len(files)} dosya okunuyor: {[os.path.basename(f) for f in files]}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    stats = {"files": [os.path.basename(f) for f in files], "rows_raw": len(df)}

    feats = [c for c in df.columns if c != LABEL]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    stats["rows_after_nan_inf"] = len(df)
    # Tekrarlar dosyalar ARASINDA da var; bu yüzden birleştirdikten sonra siliyoruz.
    df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)
    # Aynı özellik vektörü farklı etiketlerle: çoğu DoS<->DDoS flood'ları (veride doğrulandı)
    conflict = df.duplicated(subset=feats, keep=False).to_numpy()
    stats["rows_conflicting_labels"] = int(conflict.sum())
    if drop_conflicts:
        df = df[~conflict]
    stats["conflicts_dropped"] = bool(drop_conflicts)
    # Number: floodlar 100, diğer sınıflar 10 paketlik pencereyle üretilmiş -> veri toplama izi
    stats["dropped_features"] = REDUNDANT + list(drop_features)
    df = df.drop(columns=stats["dropped_features"])

    rng = np.random.default_rng(seed)
    groups = df.groupby(LABEL).indices
    keep = np.concatenate([rng.choice(ix, size=min(len(ix), cap), replace=False)
                           for ix in groups.values()])
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    stats["cap_per_fine_label"] = cap
    stats["rows_after_sampling"] = len(df)
    stats["fine_label_counts"] = {str(k): int(v) for k, v in df[LABEL].value_counts().items()}

    fine = df[LABEL].astype(str).str.upper().to_numpy()
    category = np.array([ciciot2023_category(l) for l in fine])
    X = df.drop(columns=[LABEL]).astype(np.float32)
    stats["category_counts"] = {str(k): int(v) for k, v in pd.Series(category).value_counts().items()}
    return X, fine, category, stats


# ---------------------------------------------------------------- eğitim
def fit(model, name, X, y):
    if name in NEEDS_SAMPLE_WEIGHT:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def tune(name, model, X, y, seed, n_iter, max_rows):
    if len(y) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y,
                                      random_state=seed)
    search = RandomizedSearchCV(model, PARAM_SPACE[name], n_iter=n_iter, cv=3,
                                scoring="f1_macro", random_state=seed, n_jobs=1)
    fit_params = ({"sample_weight": compute_sample_weight("balanced", y)}
                  if name in NEEDS_SAMPLE_WEIGHT else {})
    search.fit(X, y, **fit_params)
    print(f"  {name} en iyi parametreler: {search.best_params_} "
          f"(CV macro-F1={search.best_score_:.4f})")
    return search.best_params_


def run_model(name, base, Xtr, ytr, strat_tr, Xte, n_classes, args, params=None):
    # params verilirse (ör. model tohumu 1-2: seed 42'nin bulduğu ayarlar) arama YAPILMAZ.
    # Arama uzayı olmayan modellerde (TabM) arama atlanır: resmi varsayılanlar kullanılır
    if params is None:
        params = tune(name, base, Xtr, ytr, args.seed, args.n_iter, args.tune_rows) \
            if args.tune and name in PARAM_SPACE else {}
    model = clone(base).set_params(**params)

    oof = np.zeros((len(ytr), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(Xtr, strat_tr), 1):
        m = fit(clone(model), name, Xtr.iloc[tr], ytr[tr])
        oof[va] = m.predict_proba(Xtr.iloc[va])
        print(f"  {name} fold {k}/{args.folds} tamam")

    t0 = time.perf_counter()
    final = fit(clone(model), name, Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_proba = final.predict_proba(Xte).astype(np.float32)
    pred_s = time.perf_counter() - t0
    if hasattr(final, "best_iteration_"):   # early stopping'in seçtiği ağaç sayısı
        params = {**params, "n_estimators (early stopping)": final.best_iteration_}
    return {
        "params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                   for k, v in params.items()},
        "oof_proba": oof,
        "test_proba": test_proba,
        "fit_seconds": round(fit_s, 2),
        "predict_ms_per_1k": round(1000 * pred_s / len(Xte) * 1000, 3),
        "model": final,
    }


def fine_label_recall(y_true, proba, fine):
    """34 ince etiket için: kategorisi doğru tahmin edilen oran (makaledeki karışıklık analizi)."""
    hit = proba.argmax(axis=1) == y_true
    return {lab: round(float(hit[fine == lab].mean()), 6) for lab in sorted(set(fine))}


def save_shap_figures(sv, Xs, classes, name, fig_dir):
    """Makale için PNG: sınıf başına beeswarm + global ortalama |SHAP| çubuk grafiği."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    os.makedirs(fig_dir, exist_ok=True)
    for i, c in enumerate(classes):
        shap.summary_plot(sv[:, :, i], Xs, max_display=12, show=False, plot_size=(7, 5))
        plt.title(f"{name}: {c} sınıfı için SHAP değerleri", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"shap_beeswarm_{c}.png"), dpi=200)
        plt.close("all")
    imp = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([Xs.columns[j] for j in order], imp[order], color="#2a78d6", height=0.6)
    ax.set_xlabel("Ortalama |SHAP| (tüm sınıflar)")
    ax.set_title(f"{name}: global özellik önemi", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "shap_global_bar.png"), dpi=200)
    plt.close(fig)


def shap_top_features(model, name, Xte, yte, classes, per_class, seed, top_k=10,
                      fig_dir=None):
    import shap
    idx = contract.stratified_sample(yte, per_class, seed)
    Xs, ys = Xte.iloc[idx], yte[idx]
    if isinstance(model, EarlyStopping):     # sarmalayıcıyı aç; XGB'de en iyi tura kadar
        inner = model.estimator_
        model = (inner.get_booster()[: model.best_iteration_]
                 if type(inner).__name__ == "XGBClassifier" else inner)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):                 # eski shap: sınıf başına liste
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)                      # (n, özellik, sınıf)
    if fig_dir:
        save_shap_figures(sv, Xs, classes, name, fig_dir)
    sv = np.abs(sv)
    feats = list(Xte.columns)
    out = {"model": name, "n_rows": int(len(idx)), "per_class": {}}
    for i, c in enumerate(classes):
        rows = ys == i
        imp = sv[rows, :, i].mean(axis=0) if rows.any() else sv[:, :, i].mean(axis=0)
        order = np.argsort(imp)[::-1][:top_k]
        out["per_class"][c] = [{"feature": feats[j], "mean_abs_shap": round(float(imp[j]), 6)}
                               for j in order]
    glob_imp = sv.mean(axis=(0, 2))
    out["global"] = [{"feature": feats[j], "mean_abs_shap": round(float(glob_imp[j]), 6)}
                     for j in np.argsort(glob_imp)[::-1][:top_k]]
    return out


# ---------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/dataset_23")
    p.add_argument("--out-dir", default=f"outputs/{DATASET_ID}")
    p.add_argument("--work-dir", default=f"work/{DATASET_ID}",
                   help="Ara sonuçlar ve modeller (Colab'da Drive'a verin, git'e girmez)")
    p.add_argument("--cap", type=int, default=40000, help="İnce etiket başına en fazla satır")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="DT,RF,XGB,LGBM,LR",
                   help="Virgülle ayrılmış; TabM eklemek için ör. DT,RF,XGB,LGBM,LR,TabM")
    p.add_argument("--tune", action="store_true")
    # 10 deneme / 100 bin satır kararsızdı (XGB çalıştırmalar arasında 0,661 <-> 0,704)
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--tune-rows", type=int, default=200000)
    p.add_argument("--shap", action="store_true")
    p.add_argument("--shap-per-class", type=int, default=100)
    p.add_argument("--pred-per-class", type=int, default=0,
                   help="test_predictions'a sınıf başına en fazla kaç satır (0 = tüm test seti; DSS de "
                        "modellerle aynı satırlarda değerlendirilsin diye varsayılan)")
    # Resmi protokol (18.10.2026): Number atılır (veri toplama izi), çakışan etiketli
    # satırlar atılır (ekip kuralı). Duyarlılık analizi için ikisi de kapatılabilir.
    p.add_argument("--drop-features", default="Number",
                   help="Atılacak ek özellikler (virgülle). Hiçbirini atmamak için: --drop-features \"\"")
    p.add_argument("--keep-conflicts", action="store_true",
                   help="Aynı özellik + farklı etiketli satırları ATMA (duyarlılık analizi)")
    p.add_argument("--drop-conflicts", action="store_true",
                   help="Eski sürümlerle uyumluluk için; artık varsayılan")
    p.add_argument("--gpu", action="store_true", help="XGBoost'u GPU'da eğit")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42, help="Veri bölmesi, örneklem ve fold'lar (ekip protokolü: 42)")
    p.add_argument("--model-seed", type=int, default=None,
                   help="Yalnızca model eğitim tohumu (varsayılan: --seed). Bölme --seed'de sabit kalır; "
                        "--tune ile seed 42'nin bulduğu ayarlar kullanılır (arama tekrarlanmaz); çıktılar "
                        "outputs/<dataset>/seeds/s<N>'e gider, sözleşme dosyaları ezilmez.")
    p.add_argument("--quick", action="store_true", help="Küçük örneklem, 3 fold: hızlı deneme")
    p.add_argument("--force", action="store_true", help="Kaydedilmiş model sonuçlarını yok say")
    args = p.parse_args()
    args.drop_conflicts = not args.keep_conflicts
    if args.quick:
        args.cap, args.folds = min(args.cap, 3000), 3
    mseed = args.seed if args.model_seed is None else args.model_seed
    extra_seed = mseed != args.seed
    if extra_seed and os.path.normpath(args.out_dir) == os.path.normpath(f"outputs/{DATASET_ID}"):
        # Sözleşme dosyaları (DSS'in okuduğu) yalnızca resmi çalıştırmadan yazılır
        args.out_dir = os.path.join(args.out_dir, "seeds", f"s{mseed}")

    drop = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    X, fine, category, data_stats = load_data(args.data_dir, args.cap, args.seed,
                                              drop, args.drop_conflicts)
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    print(f"Örneklem: {len(y)} satır, {X.shape[1]} özellik, {len(classes)} kategori")

    # Test kilitli: ince etikete göre stratify -> nadir saldırılar da teste düşer
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=fine, random_state=args.seed)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    # Ayarlar değişince eski önbellek kullanılmasın: her ayar seti kendi klasörüne
    cfg = {k: getattr(args, k) for k in ("cap", "folds", "models", "tune", "n_iter", "tune_rows",
                                         "drop_features", "drop_conflicts", "seed", "quick")}
    # Model listesi klasörü belirlemez: her modelin sonucu kendi anahtarıyla (model_key) saklanır.
    # Liste değişince (ör. TabM eklenince) diğer modeller yeniden eğitilmesin. Sabit değer, eski
    # varsayılanla aynı tutuldu ki daha önce kaydedilmiş klasörler geçerli kalsın.
    cfg["models"] = CACHE_MODELS_KEY

    def cfg_dir(c):
        return os.path.join(args.work_dir, hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8])

    base_run_dir = cfg_dir(cfg)            # resmi (model tohumu = bölme tohumu) çalıştırma
    if extra_seed:
        # Model tohumu klasör özetine girer: seed 42'nin önbelleği okunmaz. Resmi çalıştırmanın özeti
        # değişmesin diye yalnızca tohum farklıyken eklenir.
        cfg["model_seed"] = mseed
    run_dir = cfg_dir(cfg)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        # "models" önbellek anahtarıdır, çalışan model listesi değil (o: models_run)
        json.dump({**cfg, "models_note": "önbellek anahtarı; çalışan model listesi değil",
                   "models_run": args.models}, f, indent=2, ensure_ascii=False)
    all_models = build_models(mseed, args.n_jobs, args.gpu, args.quick,
                              tabm="TabM" in args.models.split(","))
    names = [m for m in args.models.split(",") if m in all_models]
    base_models = (build_models(args.seed, args.n_jobs, args.gpu, args.quick,
                                tabm="TabM" in args.models.split(","))
                   if extra_seed and args.tune else {})
    params_from = {}
    results = {}
    for name in names:
        # Model tanımı (ör. LR max_iter) değişirse yalnızca o model yeniden eğitilir
        key = model_key(name, all_models[name])
        cache = os.path.join(run_dir, f"result_{name}_{key}.joblib")
        if os.path.exists(cache) and not args.force:
            print(f"{name}: kayıtlı sonuç kullanılıyor ({cache})")
            results[name] = joblib.load(cache)
            continue
        fixed = None
        if extra_seed and args.tune and name in PARAM_SPACE:
            # Arama tekrarlanmaz: resmi çalıştırmanın (seed 42) bulduğu ayarlar
            base_res = os.path.join(base_run_dir, f"result_{name}_{model_key(name, base_models[name])}.joblib")
            if not os.path.exists(base_res):
                raise SystemExit(f"{name}: resmi çalıştırmanın sonucu yok ({base_res}). Önce --model-seed "
                                 f"olmadan aynı ayarlarla çalıştırın.")
            fixed = {k: v for k, v in joblib.load(base_res)["params"].items()
                     if k != "n_estimators (early stopping)"}
            params_from[name] = os.path.relpath(base_res, args.work_dir)
            print(f"  {name}: seed {args.seed}'in ayarları kullanılıyor {fixed}")
        print(f"{name} eğitiliyor (model tohumu {mseed})...")
        res = run_model(name, all_models[name], Xtr, ytr, fine[idx_tr], Xte,
                        len(classes), args, params=fixed)
        joblib.dump(res.pop("model"), os.path.join(run_dir, "models", f"{name}_{key}.joblib"))
        joblib.dump(res, cache)
        results[name] = res
    model_files = {n: os.path.join(run_dir, "models", f"{n}_{model_key(n, all_models[n])}.joblib")
                   for n in names}

    metrics, oof_metrics = {}, {}
    for name, r in results.items():
        oof_metrics[name] = contract.evaluate(ytr, r["oof_proba"], classes)
        metrics[name] = {
            "test": contract.evaluate(yte, r["test_proba"], classes),
            "oof": oof_metrics[name],
            "test_fine_label_recall": fine_label_recall(yte, r["test_proba"], fine[idx_te]),
            "timing": {"fit_seconds": r["fit_seconds"],
                       "predict_ms_per_1k": r["predict_ms_per_1k"]},
            "params": r["params"],
        }
        print(f"{name:5s} OOF macro-F1={oof_metrics[name]['macro_f1']:.4f}  "
              f"test macro-F1={metrics[name]['test']['macro_f1']:.4f}")

    # Birincil model OOF'a göre seçilir, test'e göre DEĞİL
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
        shap_model = primary if primary in TREE_MODELS else next(
            (m for m in names if m in TREE_MODELS), None)
        if shap_model:
            print(f"SHAP hesaplanıyor ({shap_model})...")
            model = joblib.load(model_files[shap_model])
            shap_top = shap_top_features(model, shap_model, Xte, yte, classes,
                                         args.shap_per_class, args.seed,
                                         fig_dir=os.path.join(args.out_dir, "figures"))

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_name": "CICIoT2023 (Merged CSV)",
        "environment": "Tüketici IoT ağı (105 cihaz)",
        "owner": "Kişi 1",
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
        "model_seed": mseed,
        "params_from_seed_run": params_from or None,
        "runtime": {"gpu_used_for_xgb": bool(args.gpu), "gpu": gpu_name()},
        "data_prep": data_stats,
    }
    contract.write_outputs(args.out_dir, manifest, metrics,
                           contract.reliability_from_oof(oof_metrics), preds, shap_top)
    print(f"\nBirincil model: {primary}. Çıktılar: {args.out_dir}")
    print("Sınıf başına en iyi model (OOF F1):")
    for c, m in best_per_class.items():
        print(f"  {c:11s} {m}  F1={oof_metrics[m]['per_class'][c]['f1']:.4f}")


if __name__ == "__main__":
    main()
