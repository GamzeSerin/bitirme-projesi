"""Canlı demo paketi (Edge-IIoTset): her senaryo ailesi için, o aileyi HİÇ görmemiş TabM + kararın ihtiyaç duyduğu her şey.

Demo yerel bilgisayarda veri seti olmadan çalışacağı için gereken her şey burada bir kez hesaplanır:
  model.joblib        o aile eğitimden çıkarılmış TabM (koşul C; deneylerdeki ayarlar, tohum 42)
  meta.json           özellik sırası, bilinen sınıflar, iki eşik, skor IQR'leri, kontrol sayıları
  background.npy      Shapley tabanı: bilinen normal eğitim trafiğinden 50 satır (XAI-2 ile aynı seçim)
  grid.npy            XAI-3 araması için her özelliğin normal trafikteki yüzdelikleri (%5 … %95)
  cal_scores.npz      kalibrasyondaki bilinen trafiğin skorları (analist kartında "yüzdelik" göstermek için)

Eşikler deneylerle birebir aynı tanımla (two_signal.py / xai_unknown_decision.py):
  tau_alarm = kalibrasyondaki bilinen trafiğin üye anlaşmazlığının %95'liği
  tau_guard = kalibrasyonda alarm almamış ve BENIGN denmiş gerçek normal satırların enerjisinin (1 − q)'luğu, q = %5.
Demoda karar veren TabM'in kendisi olduğu için ana eşik TabM tahminiyle hesaplanır (tau_guard); gerçek DSS satırları varsa
DSS tahminiyle hesaplanan eşik de yazılır (tau_guard_dss) ve yeniden eğitilen modelin DSS deneyindekiyle aynı olduğu
denetlenir (sıra korelasyonu, alarm uyumu).

Çalıştırma (Kaggle GPU):
  python src/demo/export_bundle_dnn_edgeiiot.py --data-file <DNN-EdgeIIoT-dataset.csv> --params-from metrics
  → outputs/dnn_edgeiiot/demo/bundle/<AİLE>/ ve demo_bundle.zip (zip GitHub'a girmez; Kaggle çıktısından indirilir)
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import model_ml as shared
from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from open_set_lofo import model_scores

PCTS = list(range(5, 100, 10))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--params-from", choices=["run", "metrics"], default="metrics")
    p.add_argument("--families", default="WEB,BRUTEFORCE,RECON,DDOS,MALWARE")
    p.add_argument("--condition", default="C_seq_ack_yok")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--n-background", type=int, default=50)
    p.add_argument("--out-dir", default="outputs/dnn_edgeiiot/demo/bundle")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.families = 3000, "WEB"
    import dss_guard
    spec = registry.get("dnn_edgeiiot")
    manifest = json.load(open("outputs/dnn_edgeiiot/manifest.json", encoding="utf-8"))
    if args.params_from == "metrics":
        params, params_src = dss_guard.params_from_metrics("outputs/dnn_edgeiiot/metrics.json", ["TabM"])
    else:
        params, params_src = dss_guard.official_params(spec, args.run_dir, manifest, ["TabM"])
    gpu = bool(manifest.get("runtime", {}).get("gpu_used_for_xgb", False))

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, _ = registry.apply_condition(X, spec, args.condition)
    X = X.astype(np.float32)
    cols = list(X.columns)
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    benign = spec["benign"]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine, random_state=args.seed)
    print(f"Edge-IIoTset: {len(category)} satır, koşul {args.condition}, {len(cols)} özellik; TabM ayarları {params_src}")

    summary = {}
    for fam in args.families.split(","):
        t0 = time.perf_counter()
        fam_cats = spec["families"][fam]
        known = [c for c in present if c not in fam_cats]
        b = known.index(benign)
        tr = idx_tr[~np.isin(category[idx_tr], fam_cats)]
        tr_fit, tr_cal = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr], random_state=args.seed)
        yfit = np.array([known.index(c) for c in category[tr_fit]])
        Xfit, Xcal, Xte = X.iloc[tr_fit], X.iloc[tr_cal], X.iloc[idx_te]
        is_unk, is_ben = np.isin(category[idx_te], fam_cats), category[idx_te] == benign
        cal_ben = category[tr_cal] == benign

        base = shared.build_models(args.seed, -1, gpu, args.quick, tabm=True)["TabM"]
        tabm = shared.fit(clone(base).set_params(**params["TabM"]), "TabM", Xfit, yfit)
        P_cal, S_cal = model_scores("TabM", tabm, Xcal)
        P_te, S_te = model_scores("TabM", tabm, Xte)
        tau_alarm = float(np.quantile(S_cal["disagree"], args.target))
        alarm_cal, alarm_te = S_cal["disagree"] > tau_alarm, S_te["disagree"] > tau_alarm
        pred_cal, pred_te = P_cal.argmax(1), P_te.argmax(1)
        ref = ~alarm_cal & (pred_cal == b) & cal_ben
        tau_guard = float(np.quantile(S_cal["energy"][ref], 1 - args.q))

        # denetim: deneylerle aynı mı? (TabM'in kendi kararıyla iki sinyal kuralı; DSS satırlarıyla karşılaştırma)
        to_guard = ~alarm_te & (pred_te == b)
        flag = to_guard & (S_te["energy"] > tau_guard)
        check = {"silent_alarm_only": round(float((to_guard & is_unk).sum() / is_unk.sum()), 5),
                 "silent_q": round(float((to_guard & ~flag & is_unk).sum() / is_unk.sum()), 5),
                 "guard_review_q": round(float((flag & is_ben).sum() / is_ben.sum()), 5),
                 "alarm_caught": round(float((alarm_te & is_unk).sum() / is_unk.sum()), 5),
                 "alarm_review_benign": round(float((alarm_te & is_ben).sum() / is_ben.sum()), 5)}
        tau_guard_dss = None
        rows = f"outputs/dnn_edgeiiot/dss_guard/rows_{fam}_{args.condition}.npz"
        if os.path.exists(rows) and not args.quick:
            z = np.load(rows, allow_pickle=True)
            if np.array_equal(z["test_index"], idx_te) and np.array_equal(z["cal_index"], tr_cal):
                check["rank_corr_disagree_vs_dss_guard"] = round(float(spearmanr(S_te["disagree"], z["disagree_test"]).correlation), 4)
                check["rank_corr_energy_vs_dss_guard"] = round(float(spearmanr(S_te["energy"], z["energy_test"]).correlation), 4)
                check["alarm_agreement_vs_dss_guard"] = round(float((alarm_te == (z["disagree_test"] > z["tau_alarm"])).mean()), 4)
                ref_d = ~alarm_cal & (z["dss_pred_cal"] == b) & cal_ben
                tau_guard_dss = float(np.quantile(S_cal["energy"][ref_d], 1 - args.q))
            else:
                check["dss_guard_rows"] = "bölme farklı, karşılaştırılmadı"

        rng = np.random.default_rng(args.seed)
        Xfit_np = Xfit.to_numpy(np.float32)
        ben_fit = np.where(yfit == b)[0]
        background = Xfit_np[rng.choice(ben_fit, args.n_background, replace=False)]
        grid = np.percentile(Xfit_np[ben_fit], PCTS, axis=0).astype(np.float32)     # (10, M)
        iqr = {s: float(np.subtract(*np.percentile(S_cal[s], [75, 25]))) or 1.0 for s in ("disagree", "energy")}

        d = os.path.join(args.out_dir, fam)
        os.makedirs(d, exist_ok=True)
        joblib.dump(tabm, os.path.join(d, "model.joblib"))
        np.save(os.path.join(d, "background.npy"), background)
        np.save(os.path.join(d, "grid.npy"), grid)
        known_cal = ~np.isin(category[tr_cal], fam_cats)
        np.savez_compressed(os.path.join(d, "cal_scores.npz"), disagree=S_cal["disagree"][known_cal].astype(np.float32),
                            energy=S_cal["energy"][ref].astype(np.float32))
        meta = {"dataset": "Edge-IIoTset", "unseen_family": fam, "unseen_categories": fam_cats,
                "unseen_fine_labels": sorted(set(fine[idx_te][is_unk])), "condition": args.condition,
                "features": cols, "known_classes": known, "benign_index": b, "seed": args.seed,
                "target": args.target, "q": args.q, "tau_alarm": tau_alarm, "tau_guard": tau_guard,
                "tau_guard_dss": tau_guard_dss, "score_iqr": iqr, "grid_percentiles": PCTS,
                "tabm_params_source": params_src, "test_check": check,
                "rule": "alarm = disagree > tau_alarm; değilse tahmin BENIGN ve energy > tau_guard ise analiste (kontrol); "
                        "değilse tahmin edilen sınıf"}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        summary[fam] = {"seconds": round(time.perf_counter() - t0, 1), **check}
        print(f"[{fam}] {time.perf_counter() - t0:.0f} s; eşikler alarm {tau_alarm:.5f}, kontrol {tau_guard:.5f}"
              f" (DSS ile {tau_guard_dss}); denetim {check}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    z = shutil.make_archive(os.path.join(os.path.dirname(args.out_dir), "demo_bundle"), "zip", args.out_dir)
    print(f"Paket: {args.out_dir} ; zip: {z} ({os.path.getsize(z) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
