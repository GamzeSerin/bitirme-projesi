"""Gerçek DSS'te BENIGN kontrolü: görülmemiş saldırı ALLOW ediliyor mu? (ekip ortak aracı, --dataset)

two_signal.py kontrolü TabM'in kendi kararına uyguluyordu; asıl DSS ise başka modellerin birleşimiyle karar
verir. Bu araç aynı bekçiyi (TabM: alarm = üye anlaşmazlığı, kontrol = enerji) iki karara yan yana uygular:
  (a) tabm : TabM BENIGN dedi            -> two_signal.py ile aynı kurulum (karşılaştırma)
  (b) dss  : gerçek DSS BENIGN dedi      -> asıl soru

Her ailede (LOFO, ekip bölmesi seed 42; kalan eğitim %90 eğitim / %10 kalibrasyon):
  1. DSS: oylayanlar kayıttaki dss_voters (asıl DSS'le aynı, sabit). Ayarlar asıl deneyin sonuç dosyalarından
     (arama yok). Her oylayan eğitim kısmında 5-fold OOF ile eğitilir; ağırlıklar main.py'nin kuralıyla
     (sınıf başına OOF F1, sınıf başına normalize). Aksiyon main.decide ile (ciddiyet toplam olasılığı).
  2. Bekçi TabM bir kez eğitilir (oylayansa OOF'tan sonraki son modeli kullanılır).
  3. Alarm: TabM anlaşmazlığı > kalibrasyondaki bilinenlerin %95'lik dilimi -> ANALYST_REVIEW.
  4. Kontrol: alarm yok ve karar BENIGN ise enerji > tau_q -> ANALYST_REVIEW. tau_q, kalibrasyonda alarm
     almamış ve aynı kararla BENIGN denmiş GERÇEK BENIGN satırlarının (1-q) yüzdeliği. Test ve çıkarılan aile
     kullanılmaz.
Ölçütler (aile başına, sonra ortalama):
  silent_benign  görülmemiş saldırı: alarm yok, BENIGN dendi, kontrolden geçti (two_signal ile aynı tanım)
  allow          (yalnız dss) görülmemiş saldırının gerçek aksiyonu ALLOW (BENIGN + yüksek güven)
  benign_review  gerçek BENIGN test trafiğinin analiste giden oranı (maliyet; payda bütün BENIGN)
  actions        (yalnız dss) görülmemiş saldırının aksiyon dağılımı; BLOCK/ALERT = güvenli hata
Taban çizgileri: dss_only (alarm ve kontrol yok, yalnız DSS) ve alarm_only (q = 0).

Çalıştırma (asıl deneyin work dizini olan makinede; --run-dir o çalıştırmanın klasörü):
  python src/dss_guard.py --dataset dataset_23 --data-file data/dataset_23 --run-dir work/dataset_23/3ed30bf6
  python src/dss_guard.py --dataset dnn_edgeiiot --data-file <csv> --run-dir /kaggle/working/work/dnn_edgeiiot/<...>
  ... --quick           (etiket başına 3.000 satır, 3 fold, kısa eğitim: tesisat denemesi)
  ... --model-seed 1    (yalnız model tohumu; bölme --seed'de sabit, çıktı .../dss_guard/s1)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import model_ml as shared
from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES, SEVERITY
from open_set_lofo import model_scores

QS = [0.005, 0.01, 0.02, 0.05, 0.10]
ACTIONS = ["ALLOW", "LOG_MONITOR", "ANALYST_REVIEW", "ALERT", "BLOCK"]


def official_params(spec, run_dir, manifest, names):
    """Asıl deneyin seçtiği ayarlar (result_*.joblib'deki params); early stopping'in ağaç sayısı hariç,
    çünkü sarmalayıcı her eğitimde yeniden durur."""
    files = spec["model_files"](run_dir, manifest)
    missing = [n for n in names if n not in files]
    if missing:
        raise SystemExit(f"Asıl deneyde bulunamayan model(ler): {missing} (çalıştırma: {run_dir})")
    out, src = {}, {}
    for n in names:
        res = joblib.load(files[n][1])
        out[n] = {k: v for k, v in res["params"].items() if k != "n_estimators (early stopping)"}
        src[n] = os.path.basename(files[n][1])
    return out, src


def oof_and_final(name, base, X, y, strat, folds, seed, n_classes):
    """model_ml.run_model ile aynı: stratified k-fold OOF + tüm eğitim kısmında son model."""
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, strat):
        m = shared.fit(clone(base), name, X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])
    return oof, shared.fit(clone(base), name, X, y)


def dss_decide(W, probas, classes, decide):
    """main.dss_predict ile aynı: sınıf tahmini en yüksek skordan, aksiyon aynı ciddiyetteki sınıfların
    toplam olasılığından."""
    score = np.einsum("mc,mnc->nc", W, np.stack(probas))
    score = score / np.clip(score.sum(1, keepdims=True), 1e-9, None)
    sev = np.array([SEVERITY[c] for c in classes])
    top = score.argmax(1)
    sev_conf = (score * (sev[None, :] == sev[top][:, None])).sum(1)
    actions = np.array([decide(classes[j], c) for j, c in zip(top, sev_conf)])
    return top, sev_conf.astype(np.float32), actions


def action_shares(actions, mask):
    n = max(int(mask.sum()), 1)
    return {a: round(float((actions[mask] == a).sum() / n), 5) for a in ACTIONS}


def guard_eval(pred_cal, pred_te, b, alarm_cal, alarm_te, g_cal, g_te, cal_ben, is_unk, is_ben,
               dss_actions=None):
    """Bir karar kaynağı (TabM ya da DSS) için q bütçelerinde kontrolün sonucu."""
    to_guard = ~alarm_te & (pred_te == b)
    ref = ~alarm_cal & (pred_cal == b) & cal_ben
    out = []
    for q in [0.0] + QS:
        if q == 0.0:
            flag = np.zeros_like(to_guard)
            tau = None
        else:
            tau = float(np.quantile(g_cal[ref], 1 - q))
            flag = to_guard & (g_te > tau)
        r = {"q": q, "tau_guard": tau,
             "silent_benign": round(float((to_guard & ~flag & is_unk).sum() / is_unk.sum()), 5),
             "benign_review": round(float(((alarm_te | flag) & is_ben).sum() / is_ben.sum()), 5),
             "guard_review": round(float((flag & is_ben).sum() / is_ben.sum()), 5),
             "unknown_caught": round(float(((alarm_te | flag) & is_unk).sum() / is_unk.sum()), 5)}
        if dss_actions is not None:
            final = np.where(alarm_te | flag, "ANALYST_REVIEW", dss_actions)
            r["allow"] = round(float(((final == "ALLOW") & is_unk).sum() / is_unk.sum()), 5)
            r["actions_unknown"] = action_shares(final, is_unk)
            r["actions_benign"] = action_shares(final, is_ben)
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--run-dir", required=True, help="Asıl deneyin çalıştırma klasörü (ayarlar buradan)")
    p.add_argument("--manifest", default=None, help="Varsayılan outputs/<dataset>/manifest.json")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/dss_guard")
    p.add_argument("--condition", default=None, help="Varsayılan: ana koşul")
    p.add_argument("--families", default="")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42, help="Veri bölmesi ve örneklem (ekip protokolü: 42)")
    p.add_argument("--model-seed", type=int, default=None, help="Yalnız model eğitim tohumu (varsayılan --seed)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95, help="Alarm eşiği: kabul edilen bilinen oranı")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--save-rows", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.folds = 3000, 3
    spec = registry.get(args.dataset)
    args.condition = args.condition or spec["default_condition"]
    args.manifest = args.manifest or f"outputs/{args.dataset}/manifest.json"
    mseed = args.seed if args.model_seed is None else args.model_seed
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}/dss_guard"
        if mseed != args.seed:
            args.out_dir = os.path.join(args.out_dir, f"s{mseed}")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.condition if mseed == args.seed else f"{args.condition}_seed{mseed}"
    t_start = time.perf_counter()
    import main as dss_main                                  # decide() ve eşikler main.py'den

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    voters = list(spec["dss_voters"])
    names = voters + ([] if "TabM" in voters else ["TabM"])
    params, params_src = official_params(spec, args.run_dir, manifest, names)
    gpu = bool(manifest.get("runtime", {}).get("gpu_used_for_xgb", False))
    print(f"Oylayanlar (sabit): {voters}; bekçi: TabM; ayarlar: {params_src}; XGB GPU: {gpu}")

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, drop = registry.apply_condition(X, spec, args.condition)
    benign = spec["benign"]
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    fam_def = spec["families"]
    wanted = [f for f in args.families.split(",") if f] or list(fam_def)
    families = [f for f in wanted if set(fam_def[f]) & cats]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, koşul {args.condition} (çıkarılan {drop}); aileler {families}")

    results, details = [], {}
    for fam in families:
        t0 = time.perf_counter()
        fam_cats = fam_def[fam]
        known = [c for c in present if c not in fam_cats]
        b = known.index(benign)
        tr = idx_tr[~np.isin(category[idx_tr], fam_cats)]
        tr_fit, tr_cal = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr], random_state=args.seed)
        yfit = np.array([known.index(c) for c in category[tr_fit]])
        Xfit, Xcal, Xte = X.iloc[tr_fit], X.iloc[tr_cal], X.iloc[idx_te]
        is_unk = np.isin(category[idx_te], fam_cats)
        is_ben = category[idx_te] == benign
        cal_ben = category[tr_cal] == benign
        print(f"\n=== {fam} {fam_cats}: eğitim {len(tr_fit)}, kalibrasyon {len(tr_cal)}, "
              f"test bilinen {int((~is_unk).sum())}, bilinmeyen {int(is_unk.sum())} ===")

        base = shared.build_models(mseed, args.n_jobs, gpu, args.quick, tabm=True)
        W_rows, P_cal, P_te, fitted, closed = [], [], [], {}, {}
        for m in names:
            est = clone(base[m]).set_params(**params[m])
            tm = time.perf_counter()
            if m in voters:
                oof, final = oof_and_final(m, est, Xfit, yfit, fine[tr_fit], args.folds, args.seed, len(known))
                W_rows.append(f1_score(yfit, oof.argmax(1), average=None, labels=range(len(known)),
                                       zero_division=0))
                P_cal.append(final.predict_proba(Xcal))
                P_te.append(final.predict_proba(Xte))
                closed[m] = round(float(f1_score(np.array([known.index(c) for c in category[idx_te][~is_unk]]),
                                                 P_te[-1][~is_unk].argmax(1), average="macro")), 4)
            else:
                final = shared.fit(est, m, Xfit, yfit)
            fitted[m] = final
            print(f"  {m:5s} {'oylayan, OOF + son model' if m in voters else 'yalnız bekçi'} "
                  f"({time.perf_counter() - tm:.0f} s)")
        W = np.array(W_rows)
        W = W / np.clip(W.sum(0, keepdims=True), 1e-9, None)
        top_cal, _, _ = dss_decide(W, P_cal, known, dss_main.decide)
        top_te, conf_te, act_te = dss_decide(W, P_te, known, dss_main.decide)

        tabm = fitted["TabM"]
        Pt_cal, S_cal = model_scores("TabM", tabm, Xcal)
        Pt_te, S_te = model_scores("TabM", tabm, Xte)
        tau_alarm = float(np.quantile(S_cal["disagree"], args.target))
        alarm_cal, alarm_te = S_cal["disagree"] > tau_alarm, S_te["disagree"] > tau_alarm
        g_cal, g_te = S_cal["energy"], S_te["energy"]

        res_tabm = guard_eval(Pt_cal.argmax(1), Pt_te.argmax(1), b, alarm_cal, alarm_te, g_cal, g_te,
                              cal_ben, is_unk, is_ben)
        res_dss = guard_eval(top_cal, top_te, b, alarm_cal, alarm_te, g_cal, g_te, cal_ben, is_unk, is_ben,
                             dss_actions=act_te)
        for decision, rs in (("tabm", res_tabm), ("dss", res_dss)):
            for r in rs:
                results.append({"family": fam, "decision": decision, **r})
        dss_only = {"allow": round(float(((act_te == "ALLOW") & is_unk).sum() / is_unk.sum()), 5),
                    "silent_benign": round(float(((top_te == b) & is_unk).sum() / is_unk.sum()), 5),
                    "actions_unknown": action_shares(act_te, is_unk),
                    "actions_benign": action_shares(act_te, is_ben)}
        assigned = pd.Series(np.array(known)[top_te[is_unk]]).value_counts(normalize=True).round(4).to_dict()
        details[fam] = {"categories": fam_cats, "known_classes": known, "n_unknown": int(is_unk.sum()),
                        "n_benign_test": int(is_ben.sum()), "tau_alarm": tau_alarm,
                        "dss_weights": {m: dict(zip(known, np.round(W[i], 4).tolist())) for i, m in enumerate(voters)},
                        "closed_f1_known": closed, "dss_only": dss_only, "dss_unknown_assigned_to": assigned}
        q5 = {d: next(r for r in rs if r["q"] == 0.05) for d, rs in (("tabm", res_tabm), ("dss", res_dss))}
        print(f"  yalnız DSS: ALLOW {dss_only['allow']:.3f}, BENIGN {dss_only['silent_benign']:.3f} | "
              f"q=%5 -> TabM kararı: kaçış {q5['tabm']['silent_benign']:.3f}; DSS kararı: kaçış "
              f"{q5['dss']['silent_benign']:.3f}, ALLOW {q5['dss']['allow']:.3f}, inceleme "
              f"{q5['dss']['benign_review']:.3f}  ({time.perf_counter() - t0:.0f} s)")
        if args.save_rows:
            np.savez_compressed(
                os.path.join(args.out_dir, f"rows_{fam}_{tag}.npz"),
                test_index=idx_te, test_category=category[idx_te], is_unknown=is_unk,
                cal_index=tr_cal, cal_category=category[tr_cal], known=np.array(known), tau_alarm=tau_alarm,
                tabm_pred_test=Pt_te.argmax(1), tabm_pred_cal=Pt_cal.argmax(1),
                dss_pred_test=top_te, dss_pred_cal=top_cal, dss_sev_conf_test=conf_te, dss_action_test=act_te,
                disagree_test=S_te["disagree"], disagree_cal=S_cal["disagree"],
                energy_test=S_te["energy"], energy_cal=S_cal["energy"],
                msp_test=S_te["msp"], msp_cal=S_cal["msp"])

    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results])
    num = [c for c in ("silent_benign", "allow", "benign_review", "guard_review", "unknown_caught") if c in df]
    mean = df.groupby(["decision", "q"])[num].mean().round(5).reset_index()
    dss_only_mean = {k: round(float(np.mean([details[f]["dss_only"][k] for f in details])), 5)
                     for k in ("allow", "silent_benign")}
    out = {
        "dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop,
        "settings": {k: v for k, v in vars(args).items()}, "model_seed": mseed,
        "dss_voters": voters, "guard": "TabM (alarm: üye anlaşmazlığı; kontrol: enerji)",
        "params_from": params_src, "xgb_gpu": gpu,
        "guard_threshold": "kalibrasyonda alarm almamış ve aynı kararla BENIGN denmiş gerçek BENIGN satırlarının "
                           "(1-q) yüzdeliği; q = 0 yalnız alarm",
        "families": details, "results": results,
        "mean_over_families": mean.to_dict(orient="records"), "dss_only_mean": dss_only_mean,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    path = os.path.join(args.out_dir, f"dss_guard_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"\nAilelerin ortalaması (yalnız DSS: ALLOW {dss_only_mean['allow']:.4f}, "
          f"BENIGN {dss_only_mean['silent_benign']:.4f}):")
    print(mean.pivot_table(index="q", columns="decision", values=["silent_benign", "benign_review"]).round(4)
          .to_string())
    if "allow" in mean:
        print(mean[mean.decision == "dss"][["q", "allow"]].to_string(index=False))
    print(f"\nÇıktı: {path}  ({out['runtime_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
