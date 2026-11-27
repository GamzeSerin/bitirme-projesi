"""
Eğitim / doğrulama (OOF) / test skorlarını tek tabloda verir.

Kullanım (proje kök klasöründen):
    python src/train_scores.py
    python src/train_scores.py --models DT,LR      # yalnızca bazı modeller

- Eğitim skoru: kayıtlı final model, eğittiği eğitim setinde (590.515 kayıt) değerlendirilir.
- Doğrulama skoru: 5 katlı OOF (outputs/dataset_23/metrics.json'dan okunur).
- Test skoru: metrics.json'dan okunur (yalnızca seçilen model raporlanacak).
Hiçbir model yeniden eğitilmez; bölme, resmi çalıştırmayla aynıdır (seed 42, stratify=fine).
Model dosyaları kayıttaki resmi anahtarla bulunur (common/open_set_registry.py): klasörde aynı modelin
eski sürümleri dursa da resmi çalıştırmanınki kullanılır.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ml  # noqa: E402
from common import contract  # noqa: E402
from common import open_set_registry as registry  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "dataset_23"))
p.add_argument("--run-dir", default=os.path.join(ROOT, "work", "dataset_23", "3ed30bf6"))
p.add_argument("--metrics", default=os.path.join(ROOT, "outputs", "dataset_23", "metrics.json"))
p.add_argument("--manifest", default=os.path.join(ROOT, "outputs", "dataset_23", "manifest.json"))
p.add_argument("--models", default="DT,RF,XGB,LGBM,LR")
p.add_argument("--out", default=os.path.join(ROOT, "outputs", "dataset_23", "train_val_scores.json"))
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run_dir, "config.json"), encoding="utf-8"))
drop = [f.strip() for f in cfg["drop_features"].split(",") if f.strip()]
X, fine, category, _ = model_ml.load_data(a.data_dir, cfg["cap"], cfg["seed"], drop, cfg["drop_conflicts"])
classes = [c for c in model_ml.COMMON_CLASSES if c in set(category)]
y = np.array([classes.index(c) for c in category])
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=fine,
                                  random_state=cfg["seed"])
Xtr, ytr, Xte, yte = X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te], y[idx_te]
print(f"train={len(ytr)}  test={len(yte)}")

metrics = json.load(open(a.metrics, encoding="utf-8"))
benign = classes.index("BENIGN")


def dr_fpr(cm):
    cm = np.asarray(cm)
    attack = [i for i in range(len(cm)) if i != benign]
    missed = cm[attack, benign].sum()            # saldırı -> benign
    dr = 1 - missed / cm[attack].sum()
    fpr = 1 - cm[benign, benign] / cm[benign].sum()  # benign -> herhangi bir saldırı
    return round(100 * dr, 2), round(100 * fpr, 2)


rows = {}
# Resmi model dosyaları (anahtarla; eski sürümler atlanır). Bu fonksiyon model_ml'in __main__ adlarını da
# takma adla tanıtır, böylece betik olarak eğitilmiş modeller yüklenebilir.
manifest = json.load(open(a.manifest, encoding="utf-8"))
files_by_model = registry.DATASETS["dataset_23"]["model_files"](a.run_dir, manifest)
for name in a.models.split(","):
    path = files_by_model[name][0]
    files = [os.path.basename(path)]
    model = joblib.load(path)
    tr = contract.evaluate(ytr, model.predict_proba(Xtr), classes)
    te_check = contract.evaluate(yte, model.predict_proba(Xte), classes)
    m = metrics[name]
    rows[name] = {
        "model_file": files[-1],
        "train_macro_f1": tr["macro_f1"], "train_dr_fpr": dr_fpr(tr["confusion_matrix"]),
        "val_oof_macro_f1": m["oof"]["macro_f1"], "val_dr_fpr": dr_fpr(m["oof"]["confusion_matrix"]),
        "test_macro_f1": m["test"]["macro_f1"],
        "test_check_from_model": te_check["macro_f1"],   # metrics.json ile aynı olmalı
        "train_per_class_f1": {c: tr["per_class"][c]["f1"] for c in classes},
    }
    r = rows[name]
    print(f"{name:5s} train={r['train_macro_f1']:.4f}  val(OOF)={r['val_oof_macro_f1']:.4f}  "
          f"test={r['test_macro_f1']:.4f}  (kontrol {r['test_check_from_model']:.4f})  {files[-1]}")

best = max(rows, key=lambda n: rows[n]["val_oof_macro_f1"])
print(f"Doğrulama (OOF) skoruna göre seçilen model: {best}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"selected_by_validation": best, "models": rows}, open(a.out, "w", encoding="utf-8"), indent=2)
print("Kaydedildi:", a.out)
