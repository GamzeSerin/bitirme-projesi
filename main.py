"""BİRLEŞTİRME NOKTASI - Bağlama göre yönlendirmeli Karar Destek Sistemi (DSS).

Mantık:
  1) Trafik hangi ortamdan geliyorsa (dataset_id) o ortamın modellerine gider.
     Veri setlerinin özellikleri farklı olduğu için bir kayıt başka ortamın
     modeline verilmez.
  2) Ortam içinde, o ortamın modelleri sınıf bazında ağırlıklı oylanır:
         skor(c) = Σ_m w(m,c) · p_m(c)   ,  w(m,c) = OOF güvenilirliği
     Ağırlıklar yalnızca reliability.json'dan (OOF) gelir; test asla kullanılmaz.
  3) Karar katmanı: güven × ciddiyet -> aksiyon (ALLOW / LOG / REVIEW / BLOCK).

Kullanım:
  python main.py                     # tüm outputs/* değerlendir, rapor yaz
  python main.py --validate-only     # sadece sözleşme kontrolü (CI bunu çalıştırır)
  python main.py --demo dataset_23 --n 5
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import contract
from common.taxonomy import SEVERITY

OUTPUTS = "outputs"
WEIGHT_METRIC = "f1"         # reliability.json içinden: precision | recall | f1
HIGH_CONF, LOW_CONF = 0.80, 0.50
# Oylamaya yalnızca OOF macro-F1'i en iyi modelin en fazla bu kadar altında kalanlar girer.
# Kural deneyden ÖNCE sabitlendi; test sonucuna bakılarak ayarlanmaz.
VOTER_MARGIN = 0.05


class EnvironmentModel:
    """Bir veri setinin (ortamın) modelleri + güvenilirlik ağırlıkları."""

    def __init__(self, out_dir):
        self.dir = out_dir
        self.manifest = _load(os.path.join(out_dir, "manifest.json"))
        self.metrics = _load(os.path.join(out_dir, "metrics.json"))
        self.reliability = _load(os.path.join(out_dir, "reliability.json"))
        shap_path = os.path.join(out_dir, "shap_top_features.json")
        self.shap = _load(shap_path) if os.path.exists(shap_path) else None
        self.id = self.manifest["dataset_id"]
        self.classes = self.manifest["classes"]
        self.models = self.manifest["models"]
        oof = {m: self.metrics[m]["oof"]["macro_f1"] for m in self.models}
        best = max(oof.values())
        self.voters = [m for m in self.models if oof[m] >= best - VOTER_MARGIN]
        # W[m, c]: m modelinin c sınıfındaki OOF güvenilirliği, sınıf başına normalize
        W = np.array([[self.reliability[m][c][WEIGHT_METRIC] for c in self.classes]
                      for m in self.voters])
        self.W = W / np.clip(W.sum(axis=0, keepdims=True), 1e-9, None)

    def combine(self, probas):
        """probas: (model, satır, sınıf) -> (satır, sınıf) normalize skor."""
        score = np.einsum("mc,mnc->nc", self.W, probas)
        return score / np.clip(score.sum(axis=1, keepdims=True), 1e-9, None)

    def probas_from_frame(self, df, models=None):
        """(model, satır, sınıf). Varsayılan: yalnızca oylayan modeller."""
        return np.stack([df[[contract.proba_column(m, c) for c in self.classes]].to_numpy()
                         for m in (models or self.voters)])

    def explain(self, cls, k=3):
        if not self.shap:
            return []
        return [f["feature"] for f in self.shap["per_class"].get(cls, [])[:k]]


def decide(cls, confidence):
    sev = SEVERITY[cls]
    if confidence < LOW_CONF:
        return "ANALYST_REVIEW"
    if sev == 0:
        return "ALLOW" if confidence >= HIGH_CONF else "LOG_MONITOR"
    if sev >= 3 and confidence >= HIGH_CONF:
        return "BLOCK"
    if confidence >= HIGH_CONF:
        return "ALERT"
    return "LOG_MONITOR"


def dss_predict(env, probas):
    score = env.combine(probas)
    top = score.argmax(axis=1)
    out = []
    for i, j in enumerate(top):
        cls, conf = env.classes[j], float(score[i, j])
        out.append({"environment": env.id, "predicted": cls,
                    "confidence": round(conf, 4), "severity": SEVERITY[cls],
                    "action": decide(cls, conf), "explanation": env.explain(cls)})
    return out


def evaluate_environment(env):
    df = pd.read_csv(os.path.join(env.dir, "test_predictions.csv.gz"))
    y = np.array([env.classes.index(c) for c in df["y_true"]])
    all_probas = env.probas_from_frame(df, env.models)
    rows = {m: f1_score(y, all_probas[i].argmax(1), average="macro")
            for i, m in enumerate(env.models)}
    dss = dss_predict(env, env.probas_from_frame(df))
    pred = np.array([env.classes.index(d["predicted"]) for d in dss])
    rows["DSS (ağırlıklı)"] = f1_score(y, pred, average="macro")
    actions = pd.Series([d["action"] for d in dss]).value_counts().to_dict()
    # Saldırıyı ALLOW etmek en kötü hata: ayrıca raporla
    allowed_attacks = sum(1 for d, t in zip(dss, df["y_true"])
                          if d["action"] == "ALLOW" and t != "BENIGN")
    return {"macro_f1_on_test_sample": {k: round(float(v), 4) for k, v in rows.items()},
            "primary_model": env.manifest["primary_model"],
            "dss_voters": env.voters,
            "actions": {k: int(v) for k, v in actions.items()},
            "attacks_allowed": int(allowed_attacks),
            "n_rows": int(len(y))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default=OUTPUTS, help="Çıktı klasörü (varsayılan: outputs)")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--demo", metavar="DATASET_ID")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.outputs, "*")) if os.path.isdir(d))
    ready, failed = [], False
    for d in dirs:
        errors = contract.validate(d)
        if errors:
            failed = True
            print(f"✗ {d}:\n  - " + "\n  - ".join(errors))
        else:
            print(f"✓ {d}")
            ready.append(d)
    if args.validate_only:
        sys.exit(1 if failed else 0)
    if not ready:
        sys.exit("Değerlendirilecek çıktı yok. Önce src/model_*.py çalıştırın.")

    envs = {e.id: e for e in (EnvironmentModel(d) for d in ready)}

    if args.demo:
        env = envs[args.demo]
        df = pd.read_csv(os.path.join(env.dir, "test_predictions.csv.gz"))
        df = df.sample(args.n, random_state=0)
        for d, t in zip(dss_predict(env, env.probas_from_frame(df)), df["y_true"]):
            print(json.dumps({**d, "true": t}, ensure_ascii=False))
        return

    report = {eid: evaluate_environment(env) for eid, env in envs.items()}
    for eid, r in report.items():
        print(f"\n== {eid} ({r['n_rows']} test satırı, birincil: {r['primary_model']}, "
              f"DSS oylayanlar: {', '.join(r['dss_voters'])})")
        for m, f in r["macro_f1_on_test_sample"].items():
            print(f"  {m:16s} macro-F1={f:.4f}")
        print(f"  Aksiyonlar: {r['actions']}  | ALLOW edilen saldırı: {r['attacks_allowed']}")
    with open(os.path.join(args.outputs, "dss_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRapor: {os.path.join(args.outputs, 'dss_report.json')}")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
