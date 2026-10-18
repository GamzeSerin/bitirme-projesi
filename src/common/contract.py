"""Ekip içi çıktı sözleşmesi (output contract).

Herkes kendi veri setini istediği gibi işler, ama sonunda
outputs/<dataset_id>/ altına AYNI formatta şu dosyaları yazar:

    manifest.json              veri seti ve deney bilgisi
    metrics.json               model başına test + OOF metrikleri
    reliability.json           model x sınıf güvenilirlik ağırlıkları (OOF'tan!)
    test_predictions.csv.gz    test setinin sınıf başına örneklemi: y_true + olasılıklar
    shap_top_features.json     (opsiyonel) sınıf başına en önemli özellikler

main.py yalnızca bu dosyaları okur; kimsenin modeline veya verisine erişmez.
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from common.taxonomy import COMMON_CLASSES

CONTRACT_VERSION = "1.0"
REQUIRED_FILES = ["manifest.json", "metrics.json", "reliability.json",
                  "test_predictions.csv.gz"]


def proba_column(model: str, cls: str) -> str:
    return f"{model}__{cls}"


def per_class_report(y_true, y_pred, classes):
    """y_true / y_pred: classes listesindeki indeksler."""
    idx = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    total = cm.sum()
    report = {}
    for i, c in enumerate(classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        report[c] = {
            "precision": round(float(p[i]), 6),
            "recall": round(float(r[i]), 6),
            "f1": round(float(f[i]), 6),
            "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
            "support": int(s[i]),
        }
    return report, cm


def evaluate(y_true, proba, classes):
    y_pred = np.asarray(proba).argmax(axis=1)
    per_class, cm = per_class_report(y_true, y_pred, classes)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def reliability_from_oof(oof_metrics):
    """DSS ağırlıkları: model -> sınıf -> {precision, recall, f1}. Sadece OOF'tan."""
    return {
        model: {c: {k: m["per_class"][c][k] for k in ("precision", "recall", "f1")}
                for c in m["per_class"]}
        for model, m in oof_metrics.items()
    }


def stratified_sample(y, per_class, seed):
    """Her sınıftan en fazla per_class örnek: test tahmin dosyasını küçük tutar."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def environment_info():
    """Makaledeki 'deney ortamı' bölümü için: donanım + kütüphane sürümleri."""
    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "shap",
                "torch", "tensorflow"):
        try:
            libs[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "in_colab": "google.colab" in sys.modules,
        "libraries": libs,
    }


def write_outputs(out_dir, manifest, metrics, reliability, test_predictions,
                  shap_top=None):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {**manifest, "contract_version": CONTRACT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runtime": {**environment_info(), **manifest.get("runtime", {})}}
    _dump(os.path.join(out_dir, "manifest.json"), manifest)
    _dump(os.path.join(out_dir, "metrics.json"), metrics)
    _dump(os.path.join(out_dir, "reliability.json"), reliability)
    test_predictions.to_csv(os.path.join(out_dir, "test_predictions.csv.gz"),
                            index=False, float_format="%.5f", compression="gzip")
    if shap_top is not None:
        _dump(os.path.join(out_dir, "shap_top_features.json"), shap_top)
    errors = validate(out_dir)
    if errors:
        raise ValueError("Sözleşme hatası:\n" + "\n".join(errors))


def validate(out_dir):
    """Bir outputs/<dataset_id> klasörünü kontrol eder; hata listesi döner."""
    errors = [f"{f} eksik" for f in REQUIRED_FILES
              if not os.path.exists(os.path.join(out_dir, f))]
    if errors:
        return errors
    manifest = _load(os.path.join(out_dir, "manifest.json"))
    metrics = _load(os.path.join(out_dir, "metrics.json"))
    reliability = _load(os.path.join(out_dir, "reliability.json"))
    for key in ("dataset_id", "classes", "models", "primary_model"):
        if key not in manifest:
            errors.append(f"manifest.json: '{key}' alanı yok")
    if errors:
        return errors
    classes, models = manifest["classes"], manifest["models"]
    unknown = [c for c in classes if c not in COMMON_CLASSES]
    if unknown:
        errors.append(f"Ortak taksonomide olmayan sınıflar: {unknown} (taxonomy.py'ye ekleyin)")
    if manifest["primary_model"] not in models:
        errors.append("primary_model, models listesinde değil")
    for m in models:
        if m not in metrics or "test" not in metrics[m] or "oof" not in metrics[m]:
            errors.append(f"metrics.json: {m} için test/oof yok")
        if m not in reliability or set(reliability[m]) != set(classes):
            errors.append(f"reliability.json: {m} sınıfları manifest ile uyuşmuyor")
    preds = pd.read_csv(os.path.join(out_dir, "test_predictions.csv.gz"), nrows=5)
    expected = ["y_true"] + [proba_column(m, c) for m in models for c in classes]
    missing = [c for c in expected if c not in preds.columns]
    if missing:
        errors.append(f"test_predictions.csv.gz eksik sütunlar: {missing[:5]}...")
    return errors


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
