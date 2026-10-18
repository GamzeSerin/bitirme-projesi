"""Kişi 3 - dataset_26 - ŞABLON.

Kendi veri setinizi ve algoritmalarınızı istediğiniz gibi kullanın. Tek şart:
sonunda contract.write_outputs(...) ile outputs/dataset_26/ altına yazmak.
Tam çalışan bir örnek için src/model_ml.py'ye bakın.

Uyulması gereken ortak kurallar (README'deki "Ortak kurallar" bölümü):
  - Etiketler src/common/taxonomy.py içindeki ortak kategorilere eşlenir
  - Temizlik + tekrar silme split'ten ÖNCE, stratified 80/20, seed=42
  - Ağırlıklar OOF (cross-validation) tahminlerinden; test sadece en sonda
  - Her model predict_proba vermeli (SVM -> CalibratedClassifierCV)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from common import contract
from common.taxonomy import COMMON_CLASSES

DATASET_ID = "dataset_26"


def load_data(data_dir):
    """TODO: veri setini oku, temizle, X (DataFrame) ve ortak kategori dizisini döndür."""
    raise NotImplementedError


def main():
    X, category = load_data(f"data/{DATASET_ID}")
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])

    # TODO: stratified split, her model için OOF olasılıkları ve test olasılıkları
    # results = {"MLP": {"oof_proba": ..., "test_proba": ..., "fit_seconds": ..., "predict_ms_per_1k": ...}}
    results, ytr, yte = {}, None, None

    metrics, oof_metrics = {}, {}
    for name, r in results.items():
        oof_metrics[name] = contract.evaluate(ytr, r["oof_proba"], classes)
        metrics[name] = {"test": contract.evaluate(yte, r["test_proba"], classes),
                         "oof": oof_metrics[name],
                         "timing": {"fit_seconds": r["fit_seconds"],
                                    "predict_ms_per_1k": r["predict_ms_per_1k"]}}
    primary = max(oof_metrics, key=lambda m: oof_metrics[m]["macro_f1"])

    sample = contract.stratified_sample(yte, 2000, seed=42)
    preds = pd.DataFrame({"y_true": [classes[i] for i in yte[sample]]})
    for name, r in results.items():
        for i, c in enumerate(classes):
            preds[contract.proba_column(name, c)] = r["test_proba"][sample, i]

    manifest = {"dataset_id": DATASET_ID, "dataset_name": "TODO", "environment": "TODO",
                "owner": "Kişi 3", "label_level": "category", "classes": classes,
                "features": list(X.columns), "models": list(results),
                "primary_model": primary, "n_train": int(len(ytr)), "n_test": int(len(yte))}
    contract.write_outputs(f"outputs/{DATASET_ID}", manifest, metrics,
                           contract.reliability_from_oof(oof_metrics), preds)


if __name__ == "__main__":
    main()
