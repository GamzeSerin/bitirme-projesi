"""XAI-2: bekçinin (TabM) kararının açıklanması ve açıklamanın sadakati (ekip ortak aracı, --dataset).

Soru: TabM bekçisi (alarm = üye anlaşmazlığı, kontrol = enerji) bir trafiği neden şüpheli buluyor ve bu açıklama
doğru mu? Analiz planı sonuçlardan önce yazıldı (data/dnn_edgeiiot/README.md, "XAI-2"); başarı eşiği yok, aşağıdaki
ölçümlerin tamamı kaydedilir.

Her ailede (LOFO; bölme dss_guard.py ile aynı: ekip bölmesi seed 42, kalan eğitim %90 eğitim / %10 kalibrasyon):
  1. TabM bir kez eğitilir ve kaydedilir (ayarlar asıl deneyden, dss_guard ile aynı). dss_guard'ın satır dosyası
     varsa skorlar onunla karşılaştırılır (sıra korelasyonu) ve gruplar DSS'in kararıyla kurulur; yoksa TabM'in kararıyla.
  2. Gruplar (test; her birinde en fazla --per-group satır, rastgele):
       alarm_unknown   görülmemiş saldırı, alarm aldı                        -> anlaşmazlık açıklanır
       guard_unknown   görülmemiş, alarm yok, BENIGN dendi, kontrolde yakalandı -> enerji
       silent_unknown  görülmemiş, alarm yok, BENIGN dendi, kontrolden geçti    -> karşılaştırma (sadakat yok)
       alarm_benign    normal trafik, alarmla analiste                         -> anlaşmazlık
       guard_benign    normal trafik, kontrolle analiste                       -> enerji
  3. Permütasyon Shapley değerleri (kütüphanesiz; antitetik permütasyonlar): bir satırın skoru ile bilinen NORMAL
     eğitim trafiğinden (BENIGN) alınan B taban satırının ortalama skoru arasındaki farkı özelliklere paylaştırır
     ("bu satırı normal trafikten ne ayırıyor?"). İki skor aynı model çağrılarından birlikte hesaplanır.
  4. Sadakat (silme eğrisi): satırı şüpheli yapan skor için Shapley değeri en büyük k özellik bilinen normal trafikten
     alınan değerlerle değiştirilir (R taban satırı üzerinden ortalama); k = 1..K için skordaki düşüş. Ana ölçek:
     düşüş / skorun kalibrasyondaki bilinen satırlarda çeyrekler arası açıklığı (IQR) -> skordan bağımsız, sağlam.
     Ek: göreli düşüş (düşüş / (satır skoru - taban ortalaması)), yalnız paydası IQR'nin %10'undan büyük satırlarda.
     Aynısı rastgele k özellikle (--n-random tekrar). Eğri altı alan (k üzerinden ortalama), satır başına eşleştirilmiş
     fark (SHAP - rastgele), SHAP'ın daha iyi olduğu satır oranı ve Wilcoxon işaretli sıra testi.
  5. Yan bulgu: toplam |Shapley| içinde port özelliklerinin payı (kayıttaki port_columns).

Çalıştırma:
  python src/xai_guard.py --dataset dnn_edgeiiot --data-file <csv> --params-from metrics --families BRUTEFORCE,WEB
  python src/xai_guard.py --dataset dataset_23 --data-file data/dataset_23 --run-dir work/dataset_23/<...> \\
      --families RECON,WEB,BRUTEFORCE
  ... --quick   (küçük örneklem, kısa eğitim, az permütasyon: tesisat denemesi)
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
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import model_ml as shared
from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from open_set_lofo import model_scores

SCORES = ["disagree", "energy"]
GROUPS = {"alarm_unknown": "disagree", "guard_unknown": "energy", "silent_unknown": None,
          "alarm_benign": "disagree", "guard_benign": "energy"}


# ---------------------------------------------------------------- skor fonksiyonu
class ScoreFn:
    """X (n, M) numpy -> {"disagree": (n,), "energy": (n,)}; büyük partilerle."""

    def __init__(self, model, columns, batch=65536):
        self.model, self.columns, self.batch = model, list(columns), batch
        self.calls = 0

    def __call__(self, X):
        out = {s: [] for s in SCORES}
        for i in range(0, len(X), self.batch):
            _, S = model_scores("TabM", self.model, pd.DataFrame(X[i:i + self.batch], columns=self.columns))
            for s in SCORES:
                out[s].append(np.asarray(S[s], dtype=np.float64))
        self.calls += len(X)
        return {s: np.concatenate(v) for s, v in out.items()}


def permutation_shap(f, x, background, n_perm, rng):
    """Permütasyon Shapley değerleri (antitetik): (M,) her skor için.
    Her permütasyonda özellikler sırayla taban satırlarına x'in değeriyle yazılır; her adımın ortalama skor farkı
    o özelliğin marjinal katkısıdır. Toplam = f(x) - ortalama f(taban) (verimlilik)."""
    M, B = len(x), len(background)
    phi = {s: np.zeros(M) for s in SCORES}
    perms = []
    for _ in range(max(1, n_perm // 2)):
        p = rng.permutation(M)
        perms += [p, p[::-1]]
    Z = np.repeat(background[None, None, :, :], M + 1, axis=1).repeat(len(perms), axis=0)  # (P, M+1, B, M)
    for pi, p in enumerate(perms):
        for j in range(1, M + 1):
            Z[pi, j:, :, p[j - 1]] = x[p[j - 1]]
    S = f(Z.reshape(-1, M))                                              # tek parti
    for s in SCORES:
        v = S[s].reshape(len(perms), M + 1, B).mean(2)
        for pi, p in enumerate(perms):
            phi[s][p] += np.diff(v[pi])
    return {s: phi[s] / len(perms) for s in SCORES}


def deletion_curves(f, x, orders, background, ks):
    """Her sıra için ilk k özelliği taban değerleriyle değiştir; (len(orders), len(ks)) ortalama skor (tek parti)."""
    R, M = len(background), len(x)
    Z = np.empty((len(orders), len(ks), R, M))
    for o, order in enumerate(orders):
        for i, k in enumerate(ks):
            Z[o, i] = x
            Z[o, i][:, order[:k]] = background[:, order[:k]]
    S = f(Z.reshape(-1, M))
    return {s: S[s].reshape(len(orders), len(ks), R).mean(2) for s in SCORES}


def percentile_of(value, ref):
    return float((ref < value).mean() * 100)


# ---------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--run-dir", default=None, help="Asıl deneyin çalıştırma klasörü (ayarlar)")
    p.add_argument("--params-from", choices=["run", "metrics"], default="run")
    p.add_argument("--manifest", default=None)
    p.add_argument("--rows-dir", default=None, help="dss_guard satır dosyaları (varsayılan outputs/<dataset>/dss_guard)")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/xai_guard")
    p.add_argument("--condition", default=None)
    p.add_argument("--families", default="BRUTEFORCE,WEB")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=None)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95, help="Alarm eşiği (dss_guard ile aynı)")
    p.add_argument("--q", type=float, default=0.05, help="Kontrol bütçesi")
    p.add_argument("--per-group", type=int, default=50)
    p.add_argument("--n-background", type=int, default=50, help="Shapley taban satırı (B)")
    p.add_argument("--n-perm", type=int, default=10, help="Satır başına permütasyon (antitetik çiftler)")
    p.add_argument("--n-delete-bg", type=int, default=20, help="Silmede kullanılan taban satırı (R)")
    p.add_argument("--n-random", type=int, default=20, help="Rastgele sıra tekrarı")
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.per_group, args.n_background, args.n_perm = 3000, 5, 10, 2
        args.n_delete_bg, args.n_random, args.k_max = 5, 3, 4
    spec = registry.get(args.dataset)
    args.condition = args.condition or spec["default_condition"]
    args.manifest = args.manifest or f"outputs/{args.dataset}/manifest.json"
    args.rows_dir = args.rows_dir or f"outputs/{args.dataset}/dss_guard"
    mseed = args.seed if args.model_seed is None else args.model_seed
    args.out_dir = args.out_dir or f"outputs/{args.dataset}/xai_guard"
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    rng_global = np.random.default_rng(args.seed)
    t_start = time.perf_counter()

    import dss_guard
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.params_from == "metrics":
        params, params_src = dss_guard.params_from_metrics(f"outputs/{args.dataset}/metrics.json", ["TabM"])
    else:
        params, params_src = dss_guard.official_params(spec, args.run_dir, manifest, ["TabM"])
    gpu = bool(manifest.get("runtime", {}).get("gpu_used_for_xgb", False))

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, drop = registry.apply_condition(X, spec, args.condition)
    X = X.astype(np.float32)
    cols = list(X.columns)
    ports = [c for c in spec.get("port_columns", []) if c in cols]
    benign = spec["benign"]
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    fam_def = spec["families"]
    families = [f for f in args.families.split(",") if f and set(fam_def[f]) & cats]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, koşul {args.condition}, {len(cols)} özellik "
          f"(port: {ports}); aileler {families}; TabM ayarları {params_src}")

    out_fam = {}
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
        print(f"\n=== {fam}: eğitim {len(tr_fit)}, kalibrasyon {len(tr_cal)}, test {len(idx_te)} "
              f"(bilinmeyen {int(is_unk.sum())}) ===")

        # 1) TabM: eğit, kaydet
        base = shared.build_models(mseed, args.n_jobs, gpu, args.quick, tabm=True)["TabM"]
        tabm = shared.fit(clone(base).set_params(**params["TabM"]), "TabM", Xfit, yfit)
        joblib.dump(tabm, os.path.join(args.out_dir, "models", f"TabM_{fam}_{args.condition}.joblib"))
        f = ScoreFn(tabm, cols)
        P_cal, S_cal = model_scores("TabM", tabm, Xcal)
        P_te, S_te = model_scores("TabM", tabm, Xte)
        tau_alarm = float(np.quantile(S_cal["disagree"], args.target))
        alarm_cal, alarm_te = S_cal["disagree"] > tau_alarm, S_te["disagree"] > tau_alarm

        # karar kaynağı: dss_guard satırları (DSS) varsa onlar, yoksa TabM
        rows_path = os.path.join(args.rows_dir, f"rows_{fam}_{args.condition}.npz")
        check = {}
        if args.quick and os.path.exists(rows_path):
            print("  --quick: örneklem farklı olduğu için dss_guard satırları kullanılmıyor (gruplar TabM kararıyla)")
        if os.path.exists(rows_path) and not args.quick:
            z = np.load(rows_path, allow_pickle=True)
            if not (np.array_equal(z["test_index"], idx_te) and np.array_equal(z["cal_index"], tr_cal)):
                raise SystemExit(f"{rows_path}: bölme bu çalıştırmayla aynı değil")
            pred_te, pred_cal, source = z["dss_pred_test"], z["dss_pred_cal"], "dss"
            for s, key in (("disagree", "disagree_test"), ("energy", "energy_test")):
                check[s] = round(float(spearmanr(S_te[s], z[key]).correlation), 4)
            check["alarm_agreement"] = round(float((alarm_te == (z["disagree_test"] > z["tau_alarm"])).mean()), 4)
            print(f"  dss_guard ile karşılaştırma (yeniden eğitilen TabM): {check}")
        else:
            pred_te, pred_cal, source = P_te.argmax(1), P_cal.argmax(1), "tabm"
            print(f"  ! {rows_path} yok: gruplar TabM'in kararıyla")
        ref = ~alarm_cal & (pred_cal == b) & cal_ben
        tau_guard = float(np.quantile(S_cal["energy"][ref], 1 - args.q))
        to_guard = ~alarm_te & (pred_te == b)
        flag = to_guard & (S_te["energy"] > tau_guard)
        masks = {"alarm_unknown": alarm_te & is_unk, "guard_unknown": flag & is_unk,
                 "silent_unknown": to_guard & ~flag & is_unk,
                 "alarm_benign": alarm_te & is_ben, "guard_benign": flag & is_ben}
        counts = {g: int(m.sum()) for g, m in masks.items()}
        print(f"  gruplar ({source} kararı): {counts}")

        # taban: bilinen NORMAL eğitim trafiğinden
        rng = np.random.default_rng(args.seed)
        Xfit_np = Xfit.to_numpy(np.float32)
        ben_fit = np.where(yfit == b)[0]
        bg = Xfit_np[rng.choice(ben_fit, args.n_background, replace=False)]
        del_bg = Xfit_np[rng.choice(ben_fit, args.n_delete_bg, replace=False)]
        iqr = {s: float(np.subtract(*np.percentile(S_cal[s], [75, 25]))) or 1.0 for s in SCORES}
        bg_scores = f(bg)
        del_bg_scores = f(del_bg)
        ks = list(range(1, min(args.k_max, len(cols)) + 1))
        Xte_np = Xte.to_numpy(np.float32)

        groups_out = {}
        rows_save = {}
        for g, score in GROUPS.items():
            idx = np.where(masks[g])[0]
            if len(idx) == 0:
                groups_out[g] = {"n_available": 0}
                continue
            idx = np.sort(rng.choice(idx, min(args.per_group, len(idx)), replace=False))
            phis = {s: [] for s in SCORES}
            fx = {s: [] for s in SCORES}
            del_shap, del_rand, rel_shap, rel_rand = [], [], [], []
            for i in idx:
                x = Xte_np[i]
                ph = permutation_shap(f, x, bg, args.n_perm, rng)
                fxi = f(x[None, :])
                for s in SCORES:
                    phis[s].append(ph[s])
                    fx[s].append(float(fxi[s][0]))
                if score is None:
                    continue
                gap0 = fxi[score][0] - float(del_bg_scores[score].mean())
                order = np.argsort(-ph[score])
                orders = [order] + [rng.permutation(len(cols)) for _ in range(args.n_random)]
                D = deletion_curves(f, x, orders, del_bg, ks)[score]
                d, rr = D[0], D[1:].mean(0)
                del_shap.append((fxi[score][0] - d) / iqr[score])
                del_rand.append((fxi[score][0] - rr) / iqr[score])
                if gap0 > 0.1 * iqr[score]:
                    rel_shap.append((fxi[score][0] - d) / gap0)
                    rel_rand.append((fxi[score][0] - rr) / gap0)
            phis = {s: np.array(v) for s, v in phis.items()}
            gi = {"n_available": int(masks[g].sum()), "n_explained": int(len(idx)), "explained_score": score}
            for s in SCORES:
                a = np.abs(phis[s]).mean(0)
                order = np.argsort(-a)
                gi[f"top_{s}"] = [{"feature": cols[j], "mean_abs_shap": round(float(a[j]), 6),
                                   "mean_shap": round(float(phis[s][:, j].mean()), 6)} for j in order[:10]]
                tot = np.abs(phis[s]).sum(1)
                gi[f"port_share_{s}"] = (round(float((np.abs(phis[s][:, [cols.index(c) for c in ports]]).sum(1)
                                                      / np.clip(tot, 1e-12, None)).mean()), 4) if ports else None)
                # verimlilik kontrolü: Shapley toplamı = f(x) - taban ortalaması
                gap = np.array(fx[s]) - bg_scores[s].mean() - phis[s].sum(1)
                gi[f"efficiency_gap_{s}"] = round(float(np.abs(gap).mean()), 6)
            if score is not None:
                ds, dr = np.array(del_shap), np.array(del_rand)       # (n, K), IQR birimiyle düşüş
                auc_s, auc_r = ds.mean(1), dr.mean(1)
                diff = auc_s - auc_r
                try:
                    w = wilcoxon(diff)
                    w_stat, w_p = float(w.statistic), float(w.pvalue)
                except ValueError:
                    w_stat, w_p = None, None
                gi["faithfulness"] = {
                    "score": score, "ks": ks, "scale": "düşüş / kalibrasyon IQR", "iqr": round(iqr[score], 6),
                    "mean_drop_iqr_shap": np.round(ds.mean(0), 4).tolist(),
                    "mean_drop_iqr_random": np.round(dr.mean(0), 4).tolist(),
                    "relative": ({"n_rows": len(rel_shap),
                                  "mean_rel_drop_shap": np.round(np.mean(rel_shap, 0), 4).tolist(),
                                  "mean_rel_drop_random": np.round(np.mean(rel_rand, 0), 4).tolist()}
                                 if rel_shap else {"n_rows": 0}),
                    "auc_shap_mean": round(float(auc_s.mean()), 4), "auc_random_mean": round(float(auc_r.mean()), 4),
                    "paired_diff_median": round(float(np.median(diff)), 4),
                    "paired_diff_mean": round(float(diff.mean()), 4),
                    "frac_rows_shap_better": round(float((diff > 0).mean()), 4),
                    "wilcoxon_statistic": w_stat, "wilcoxon_p": w_p}
                print(f"  {g:15s} ({score}): n={len(idx)}  AUC SHAP {auc_s.mean():.3f} / rastgele {auc_r.mean():.3f}"
                      f"  SHAP daha iyi %{100 * (diff > 0).mean():.0f}  p={w_p}"
                      + (f"  port payı {gi[f'port_share_{score}']}" if ports else ""))
            # analist kartları: ilk 3 satır, skoru açıklayan özelliklerin en çok artıranları
            cards = []
            s_card = score or "energy"
            for r in range(min(3, len(idx))):
                top = np.argsort(-phis[s_card][r])[:3]
                cards.append({"test_row": int(idx_te[idx[r]]), "category": str(category[idx_te][idx[r]]),
                              "score": s_card, "score_value": round(fx[s_card][r], 5),
                              "reasons": [{"feature": cols[j], "value": float(Xte_np[idx[r], j]),
                                           "shap": round(float(phis[s_card][r, j]), 6),
                                           "percentile_in_known": round(percentile_of(Xte_np[idx[r], j],
                                                                                      Xfit_np[:, j]), 1)}
                                          for j in top]})
            gi["analyst_cards"] = cards
            groups_out[g] = gi
            rows_save[g] = {"idx": idx, **{f"shap_{s}": phis[s] for s in SCORES},
                            **{f"f_{s}": np.array(fx[s]) for s in SCORES}}
        np.savez_compressed(os.path.join(args.out_dir, f"shap_{fam}_{args.condition}.npz"), columns=np.array(cols),
                            **{f"{g}__{k}": v for g, d in rows_save.items() for k, v in d.items()})
        out_fam[fam] = {"categories": fam_cats, "decision_source": source, "check_vs_dss_guard": check,
                        "tau_alarm": tau_alarm, "tau_guard": tau_guard, "group_sizes": counts,
                        "background": "bilinen normal (BENIGN) eğitim trafiği",
                        "background_mean_score": {s: round(float(bg_scores[s].mean()), 6) for s in SCORES},
                        "groups": groups_out, "model_calls": f.calls,
                        "seconds": round(time.perf_counter() - t0, 1)}
        print(f"  ({time.perf_counter() - t0:.0f} s, {f.calls:,} model satırı)")

    out = {"dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop,
           "status": "Analiz planı sonuçlardan önce yazıldı (README, XAI-2); eşik yok",
           "settings": {k: v for k, v in vars(args).items()}, "model_seed": mseed, "tabm_params_from": params_src,
           "port_columns": ports, "families": out_fam, "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    path = os.path.join(args.out_dir, f"xai_guard_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plot(out, os.path.join(args.out_dir, f"fig_xai_guard_{args.dataset}_{args.condition}.png"))
    print(f"\nÇıktı: {path}  ({out['runtime_seconds']:.0f} s)")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(fam, g, gi) for fam, fd in out["families"].items() for g, gi in fd["groups"].items()
              if "faithfulness" in gi]
    if not panels:
        return
    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    for ax, (fam, g, gi) in zip(axes.flat, panels):
        fz = gi["faithfulness"]
        ax.plot(fz["ks"], fz["mean_drop_iqr_shap"], marker="o", color="#2a78d6", label="SHAP sırası")
        ax.plot(fz["ks"], fz["mean_drop_iqr_random"], marker="o", color="#9aa5b1", label="rastgele")
        ax.set_title(f"{fam} · {g}\n({fz['score']}, n={gi['n_explained']})", fontsize=8)
        ax.set_xlabel("değiştirilen özellik sayısı k", fontsize=7)
        ax.set_ylabel("skor düşüşü (IQR)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(fz["ks"])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{out['dataset_id']}: bekçi açıklamalarının sadakati (silme eğrisi)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
