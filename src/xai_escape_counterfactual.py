"""Kaçışların Karşı-Olgusal Açıklaması (XAI-3): bekçiden sessizce geçen görülmemiş saldırı satırı hangi özellikleri
ne kadar farklı olsaydı yakalanırdı? (ekip ortak aracı, --dataset)

Analiz planı sonuçlardan önce yazıldı (data/dnn_edgeiiot/README.md, "Kaçışların Karşı-Olgusal Açıklaması (XAI-3)");
başarı eşiği yok, aşağıdaki ölçümlerin tamamı kaydedilir.

Her ailede (LOFO; bölme, TabM, eşikler ve satır grupları Bilinmeyen Kararının Açıklanması (XAI-2) koşusuyla aynı):
  1. XAI-2'nin kaydettiği TabM yüklenir (yeniden eğitim yok); skorlar dss_guard satır dosyasıyla karşılaştırılır.
  2. Açıklanan satırlar: sessiz kaçanlar (alarm yok, DSS BENIGN dedi, enerji kontrolünden geçti); XAI-2'nin bu grupta
     açıkladığı satırların aynısı (XAI-2 Shapley değerleriyle karşılaştırma için).
  3. Yakalanma payı (IQR birimi): m = max((anlaşmazlık - alarm eşiği) / IQR_anl, (enerji - kontrol eşiği) / IQR_ene).
     m > 0 ise satır yakalanır (alarm veya kontrol). DSS kararı yeniden hesaplanmaz (BENIGN demeye devam ettiği varsayılır).
  4. Arama (açgözlü, kütüphanesiz): her adımda değişmemiş bir özelliğin değeri, bilinen eğitim trafiğindeki o özelliğin
     yüzdeliklerinden (%5, %15, ..., %95) biriyle değiştirilir; m'yi en çok artıran değişiklik seçilir. En fazla k-max adım.
  5. Rastgele taban: aynı adım sayısıyla rastgele özellik ve rastgele yüzdelik (--n-random tekrar).
     Ek analiz (sonuçlardan önce eklendi): aynı arama yalnız enerji kontrolü için (alarm sayılmaz), çünkü XAI-2'de
     anlaşmazlığın veride görülmemiş karışımlarla kolayca yükseldiği görüldü.
  6. Raporlanan: başlangıç payı; k = 1..K yakalanma oranı (açgözlü / rastgele); gereken özellik sayısı ve değişim
     büyüklüğü (özelliğin bilinen trafikteki IQR'si biriminde); en sık değişen özellikler ve tetiklenen sinyal; XAI-2 ile
     tutarlılık (ilk adımdaki özellik kazançları ile |Shapley| arasında sıra korelasyonu, değişen özelliklerin |Shapley|
     sırası ve ilk-5 örtüşmesi); gerçekçilik (karşı-olgunun bilinen eğitim trafiğinde en yakın komşu uzaklığı, gerçek
     bilinen satırlarınkine göre); port değişikliklerinin oranı.
  7. Eş kontrolü (sonradan önerildi, sonuçlardan önce plan README'de; 26.10.2026): görülmemiş saldırı gruplarının
     (alarm alan, kontrolde yakalanan, sessiz kaçan) TAMAMI için bilinen eğitim trafiğindeki en yakın satır (ölçekli
     uzay, tüm eğitim satırları): uzaklık, birebir eş (uzaklık 0) oranı ve en yakın eşin kategorisi.

Çalıştırma (önce xai_unknown_decision.py koşmuş olmalı):
  python src/xai_escape_counterfactual.py --dataset dnn_edgeiiot --data-file <csv> --families BRUTEFORCE,WEB
  ... --quick   (küçük örneklem; TabM kısa eğitilir, XAI-2 çıktısı kullanılmaz: tesisat denemesi)
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
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import model_ml as shared
from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from open_set_lofo import model_scores
from xai_unknown_decision import SCORES, ScoreFn

QUANTS = np.arange(5, 100, 10)            # %5, %15, ..., %95


def feature_scale(Xfit):
    """Özellik başına ölçek: IQR; sıfırsa %5-%95 aralığı; o da sıfırsa 1."""
    q = np.percentile(Xfit, [5, 25, 75, 95], axis=0)
    s = q[2] - q[1]
    s = np.where(s > 0, s, q[3] - q[0])
    return np.where(s > 0, s, 1.0)


class Margin:
    """Satır(lar)ın yakalanma payı (IQR birimi) ve hangi sinyalin önde olduğu."""

    def __init__(self, f, tau_alarm, tau_guard, iqr, energy_only=False):
        self.f, self.ta, self.tg, self.iqr, self.energy_only = f, tau_alarm, tau_guard, iqr, energy_only

    def __call__(self, Z):
        S = self.f(Z)
        ma = (S["disagree"] - self.ta) / self.iqr["disagree"]
        me = (S["energy"] - self.tg) / self.iqr["energy"]
        if self.energy_only:                 # ek analiz: yalnız enerji kontrolüyle yakalanma
            return me, np.full_like(me, -np.inf), me
        return np.maximum(ma, me), ma, me


def greedy_cf(margin, x, grid, k_max):
    """Açgözlü karşı-olgu. Döner: adımlar [(j, eski, yeni, m)], başlangıç m, ilk adım kazançları (M,), son (ma, me)."""
    M = len(x)
    cur = x.copy()
    m0 = float(margin(cur[None])[0][0])
    m_cur, steps, first_gain = m0, [], None
    free = np.ones(M, bool)
    for _ in range(k_max):
        cand = [(j, v) for j in np.where(free)[0] for v in grid[j] if v != cur[j]]
        if not cand:
            break
        Z = np.repeat(cur[None], len(cand), 0)
        for r, (j, v) in enumerate(cand):
            Z[r, j] = v
        m, _, _ = margin(Z)
        if first_gain is None:
            first_gain = np.full(M, -np.inf)
            for r, (j, _) in enumerate(cand):
                first_gain[j] = max(first_gain[j], m[r] - m0)
        best = int(np.argmax(m))
        j, v = cand[best]
        if m[best] <= m_cur:                 # hiçbir tek değişiklik payı artırmıyor
            break
        steps.append((int(j), float(cur[j]), float(v), float(m[best])))
        cur[j] = v
        free[j] = False
        m_cur = float(m[best])
        if m_cur > 0:
            break
    _, ma, me = margin(cur[None])
    return steps, m0, first_gain, cur, (float(ma[0]), float(me[0]))


def random_caught_curve(margin, x, grid, k_max, n_rep, rng):
    """Rastgele özellik + rastgele yüzdelik; (n_rep, k_max) 'k adımda (veya önce) yakalandı mı'."""
    M = len(x)
    Z = np.empty((n_rep, k_max, M), dtype=x.dtype)
    for r in range(n_rep):
        cur = x.copy()
        for k, j in enumerate(rng.permutation(M)[:k_max]):
            cur[j] = grid[j][rng.integers(len(grid[j]))]
            Z[r, k] = cur
    m, _, _ = margin(Z.reshape(-1, M))
    caught = (m.reshape(n_rep, k_max) > 0)
    return np.maximum.accumulate(caught, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--run-dir", default=None, help="(yalnız --quick) asıl deneyin klasörü, TabM ayarları için")
    p.add_argument("--params-from", choices=["run", "metrics"], default="metrics")
    p.add_argument("--manifest", default=None)
    p.add_argument("--rows-dir", default=None, help="dss_guard satır dosyaları (varsayılan outputs/<dataset>/dss_guard)")
    p.add_argument("--xai2-dir", default=None, help="XAI-2 çıktısı (varsayılan outputs/<dataset>/xai_unknown_decision)")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/xai_escape_counterfactual")
    p.add_argument("--condition", default=None)
    p.add_argument("--families", default="BRUTEFORCE,WEB")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--per-group", type=int, default=50)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-real-ref", type=int, default=1000, help="Gerçekçilik için gerçek bilinen satır sayısı")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.per_group, args.n_random, args.n_real_ref = 3000, 5, 3, 200
    spec = registry.get(args.dataset)
    args.condition = args.condition or spec["default_condition"]
    args.manifest = args.manifest or f"outputs/{args.dataset}/manifest.json"
    args.rows_dir = args.rows_dir or f"outputs/{args.dataset}/dss_guard"
    args.xai2_dir = args.xai2_dir or f"outputs/{args.dataset}/xai_unknown_decision"
    args.out_dir = args.out_dir or f"outputs/{args.dataset}/xai_escape_counterfactual"
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()

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
          f"(port: {ports}); aileler {families}")

    if args.quick:
        import dss_guard
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if args.params_from == "metrics":
            params, _ = dss_guard.params_from_metrics(f"outputs/{args.dataset}/metrics.json", ["TabM"])
        else:
            params, _ = dss_guard.official_params(spec, args.run_dir, manifest, ["TabM"])
        gpu = bool(manifest.get("runtime", {}).get("gpu_used_for_xgb", False))

    out_fam, save = {}, {}
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
        cal_ben = category[tr_cal] == benign
        print(f"\n=== {fam}: eğitim {len(tr_fit)}, kalibrasyon {len(tr_cal)}, test {len(idx_te)} "
              f"(bilinmeyen {int(is_unk.sum())}) ===")

        # 1) TabM: XAI-2'nin kaydettiği model (quick'te kısa eğitim)
        if args.quick:
            base = shared.build_models(args.seed, args.n_jobs, gpu, True, tabm=True)["TabM"]
            tabm = shared.fit(clone(base).set_params(**params["TabM"]), "TabM", Xfit, yfit)
            model_src = "quick: yeniden eğitildi"
        else:
            mp = os.path.join(args.xai2_dir, "models", f"TabM_{fam}_{args.condition}.joblib")
            if not os.path.exists(mp):
                raise SystemExit(f"{mp} yok: önce xai_unknown_decision.py koşmalı")
            tabm = joblib.load(mp)
            model_src = mp
        f = ScoreFn(tabm, cols)
        P_cal, S_cal = model_scores("TabM", tabm, Xcal)
        P_te, S_te = model_scores("TabM", tabm, Xte)
        tau_alarm = float(np.quantile(S_cal["disagree"], args.target))
        alarm_cal, alarm_te = S_cal["disagree"] > tau_alarm, S_te["disagree"] > tau_alarm

        rows_path = os.path.join(args.rows_dir, f"rows_{fam}_{args.condition}.npz")
        check = {}
        if not args.quick:
            if not os.path.exists(rows_path):
                raise SystemExit(f"{rows_path} yok: gruplar DSS kararıyla kurulmalı")
            z = np.load(rows_path, allow_pickle=True)
            if not (np.array_equal(z["test_index"], idx_te) and np.array_equal(z["cal_index"], tr_cal)):
                raise SystemExit(f"{rows_path}: bölme bu çalıştırmayla aynı değil")
            pred_te, pred_cal, source = z["dss_pred_test"], z["dss_pred_cal"], "dss"
            for s, key in (("disagree", "disagree_test"), ("energy", "energy_test")):
                check[s] = round(float(spearmanr(S_te[s], z[key]).correlation), 4)
            check["alarm_agreement"] = round(float((alarm_te == (z["disagree_test"] > z["tau_alarm"])).mean()), 4)
            print(f"  dss_guard ile karşılaştırma (kayıtlı TabM): {check}")
        else:
            pred_te, pred_cal, source = P_te.argmax(1), P_cal.argmax(1), "tabm"
            print("  --quick: gruplar TabM kararıyla, XAI-2 çıktısı kullanılmıyor")
        ref = ~alarm_cal & (pred_cal == b) & cal_ben
        tau_guard = float(np.quantile(S_cal["energy"][ref], 1 - args.q))
        to_guard = ~alarm_te & (pred_te == b)
        flag = to_guard & (S_te["energy"] > tau_guard)
        silent = to_guard & ~flag & is_unk
        print(f"  sessiz kaçan: {int(silent.sum())}")
        if not silent.any():
            out_fam[fam] = {"n_silent": 0}
            continue

        # 2) satırlar: XAI-2'nin açıkladığı sessiz kaçanlar (Shapley ile)
        rng = np.random.default_rng(args.seed)
        shap = None
        if not args.quick:
            sp = os.path.join(args.xai2_dir, f"shap_{fam}_{args.condition}.npz")
            zs = np.load(sp, allow_pickle=True)
            if list(zs["columns"]) != cols:
                raise SystemExit(f"{sp}: sütunlar farklı")
            idx = zs["silent_unknown__idx"]
            if not silent[idx].all():
                raise SystemExit(f"{sp}: sessiz kaçan satırlar bu çalıştırmayla aynı değil")
            shap = {s: zs[f"silent_unknown__shap_{s}"] for s in SCORES}
        else:
            idx = np.sort(rng.choice(np.where(silent)[0], min(args.per_group, int(silent.sum())), replace=False))
        if len(idx) == 0:
            out_fam[fam] = {"n_silent": 0}
            continue

        Xfit_np = Xfit.to_numpy(np.float32)
        Xte_np = Xte.to_numpy(np.float32)
        grid = [np.unique(np.percentile(Xfit_np[:, j], QUANTS).astype(np.float32)) for j in range(len(cols))]
        scale = feature_scale(Xfit_np)
        iqr = {s: float(np.subtract(*np.percentile(S_cal[s], [75, 25]))) or 1.0 for s in SCORES}
        margin = Margin(f, tau_alarm, tau_guard, iqr)
        margin_e = Margin(f, tau_alarm, tau_guard, iqr, energy_only=True)

        # gerçekçilik: bilinen eğitim trafiğinde en yakın komşu (ölçekli uzay)
        Xs = Xfit_np / scale
        ref_pool = Xs if len(Xs) <= 100000 else Xs[rng.choice(len(Xs), 100000, replace=False)]
        nn = NearestNeighbors(n_neighbors=2).fit(ref_pool)
        real = ref_pool[rng.choice(len(ref_pool), min(args.n_real_ref, len(ref_pool)), replace=False)]
        d_real = nn.kneighbors(real)[0][:, 1]          # kendisi hariç en yakın

        rows, rand_curves, rand_curves_e = [], [], []
        ks = list(range(1, args.k_max + 1))
        for r, i in enumerate(idx):
            x = Xte_np[i].copy()
            steps, m0, gain, cf, (ma, me) = greedy_cf(margin, x, grid, args.k_max)
            caught = bool(steps) and steps[-1][3] > 0
            rc = random_caught_curve(margin, x, grid, args.k_max, args.n_random, rng)
            rand_curves.append(rc.mean(0))
            row = {"test_row": int(idx_te[i]), "m0": m0,
                   "m0_alarm": float(margin(x[None])[1][0]), "m0_energy": float(margin(x[None])[2][0]),
                   "caught": caught, "k_needed": len(steps) if caught else None,
                   "trigger": (("alarm" if ma > 0 else "") + ("+" if ma > 0 and me > 0 else "")
                               + ("energy" if me > 0 else "")) if caught else None,
                   "changes": [{"feature": cols[j], "from": a, "to": v, "size_iqr": abs(v - a) / float(scale[j]),
                                "m_after": m} for j, a, v, m in steps],
                   "d_nn_original": float(nn.kneighbors((x / scale)[None], n_neighbors=1)[0][0, 0]),
                   "d_nn_cf": float(nn.kneighbors((cf / scale)[None], n_neighbors=1)[0][0, 0])}
            if shap is not None:
                s_key = "disagree" if (caught and ma > 0 and ma >= me) else "energy"
                a = np.abs(shap[s_key][r])
                rank = np.empty(len(cols), int)
                rank[np.argsort(-a)] = np.arange(1, len(cols) + 1)
                ok = np.isfinite(gain)
                row["consistency"] = {"score": s_key,
                                      "spearman_gain_vs_abs_shap": float(spearmanr(gain[ok], a[ok]).correlation),
                                      "shap_rank_of_changed": [int(rank[j]) for j, *_ in steps],
                                      "changed_in_shap_top5": [bool(rank[j] <= 5) for j, *_ in steps]}
            # ek analiz (sonuçlardan önce eklendi): yalnız enerji kontrolüyle yakalanma
            st_e, _, _, _, _ = greedy_cf(margin_e, x, grid, args.k_max)
            ce = bool(st_e) and st_e[-1][3] > 0
            row["energy_only"] = {"caught": ce, "k_needed": len(st_e) if ce else None,
                                  "changes": [{"feature": cols[j], "from": a, "to": v,
                                               "size_iqr": abs(v - a) / float(scale[j]), "m_after": m}
                                              for j, a, v, m in st_e]}
            rand_curves_e.append(random_caught_curve(margin_e, x, grid, args.k_max, args.n_random, rng).mean(0))
            rows.append(row)

        # 3) özet
        n = len(rows)
        caught = np.array([r["caught"] for r in rows])
        k_need = np.array([r["k_needed"] if r["caught"] else np.inf for r in rows])
        greedy_curve = [float((k_need <= k).mean()) for k in ks]
        random_curve = np.mean(rand_curves, 0).tolist()
        changes = [c for r in rows if r["caught"] for c in r["changes"]]
        freq = {}
        for c in changes:
            freq[c["feature"]] = freq.get(c["feature"], 0) + 1
        top_feat = sorted(freq.items(), key=lambda t: -t[1])
        trig = {}
        for r in rows:
            if r["caught"]:
                trig[r["trigger"]] = trig.get(r["trigger"], 0) + 1
        first = [r["changes"][0]["feature"] for r in rows if r["caught"] and r["changes"]]
        m0 = np.array([r["m0"] for r in rows])
        d_cf = np.array([r["d_nn_cf"] for r in rows if r["caught"]])
        d_org = np.array([r["d_nn_original"] for r in rows])
        summ = {
            "n_silent": int(silent.sum()), "n_explained": n, "rows_from": "XAI-2 sessiz kaçan örneklemi" if shap is not None
            else "quick: rastgele",
            "margin_start": {"median": float(np.median(m0)), "q10": float(np.quantile(m0, 0.1)),
                             "q90": float(np.quantile(m0, 0.9)),
                             "median_alarm": float(np.median([r["m0_alarm"] for r in rows])),
                             "median_energy": float(np.median([r["m0_energy"] for r in rows]))},
            "caught_curve": {"ks": ks, "greedy": greedy_curve, "random": [round(v, 4) for v in random_curve]},
            "k_needed_median": (float(np.median(k_need[caught])) if caught.any() else None),
            "change_size_iqr_median": (float(np.median([c["size_iqr"] for c in changes])) if changes else None),
            "change_size_iqr_q90": (float(np.quantile([c["size_iqr"] for c in changes], 0.9)) if changes else None),
            "top_changed_features": [{"feature": k, "count": v, "share_of_changes": round(v / len(changes), 4)}
                                     for k, v in top_feat[:10]],
            "first_change_features": {k: first.count(k) for k in sorted(set(first), key=lambda t: -first.count(t))},
            "trigger": trig,
            "port_share_of_changes": (round(sum(c["feature"] in ports for c in changes) / len(changes), 4)
                                      if changes and ports else None),
            "rows_with_port_change": (round(float(np.mean([any(c["feature"] in ports for c in r["changes"])
                                                           for r in rows if r["caught"]])), 4)
                                      if caught.any() and ports else None),
            "realism": {"d_nn_real_known_median": float(np.median(d_real)),
                        "d_nn_original_median": float(np.median(d_org)),
                        "d_nn_cf_median": (float(np.median(d_cf)) if len(d_cf) else None),
                        "cf_over_real_median_ratio": (float(np.median(d_cf) / max(np.median(d_real), 1e-12))
                                                      if len(d_cf) else None),
                        "cf_percentile_in_real": ([round(float((d_real < d).mean() * 100), 1) for d in d_cf]
                                                  if len(d_cf) else [])},
        }
        ke = np.array([r["energy_only"]["k_needed"] if r["energy_only"]["caught"] else np.inf for r in rows])
        ch_e = [c for r in rows if r["energy_only"]["caught"] for c in r["energy_only"]["changes"]]
        fe = {}
        for c in ch_e:
            fe[c["feature"]] = fe.get(c["feature"], 0) + 1
        summ["energy_only"] = {
            "note": "ek analiz, sonuçlardan önce eklendi: yalnız enerji kontrolü (alarm sayılmaz)",
            "caught_curve": {"ks": ks, "greedy": [float((ke <= k).mean()) for k in ks],
                             "random": [round(v, 4) for v in np.mean(rand_curves_e, 0).tolist()]},
            "top_changed_features": [{"feature": k, "count": v, "share_of_changes": round(v / len(ch_e), 4)}
                                     for k, v in sorted(fe.items(), key=lambda t: -t[1])[:10]],
            "port_share_of_changes": (round(sum(c["feature"] in ports for c in ch_e) / len(ch_e), 4)
                                      if ch_e and ports else None),
            "change_size_iqr_median": (float(np.median([c["size_iqr"] for c in ch_e])) if ch_e else None)}
        if shap is not None:
            cons = [r["consistency"] for r in rows]
            ranks = [k for c, r in zip(cons, rows) if r["caught"] for k in c["shap_rank_of_changed"]]
            summ["consistency_with_xai2"] = {
                "spearman_gain_vs_abs_shap_median": float(np.nanmedian([c["spearman_gain_vs_abs_shap"] for c in cons])),
                "shap_rank_of_changed_median": (float(np.median(ranks)) if ranks else None),
                "changed_in_shap_top5_share": (float(np.mean([v for c, r in zip(cons, rows) if r["caught"]
                                                               for v in c["changed_in_shap_top5"]])) if ranks else None)}
        print(f"  n={n}  başlangıç payı medyan {summ['margin_start']['median']:.2f} IQR  "
              f"yakalanma k=1..{args.k_max}: {[round(v, 2) for v in greedy_curve]} "
              f"(rastgele {[round(v, 2) for v in random_curve]})")
        print(f"  en sık değişen: {top_feat[:5]}  tetik: {trig}  port payı: {summ['port_share_of_changes']}")
        eo = summ["energy_only"]
        print(f"  yalnız enerji: yakalanma {[round(v, 2) for v in eo['caught_curve']['greedy']]} "
              f"(rastgele {eo['caught_curve']['random']})  en sık {[(t['feature'], t['count']) for t in eo['top_changed_features'][:5]]}"
              f"  port payı {eo['port_share_of_changes']}")
        if shap is not None:
            print(f"  XAI-2 ile tutarlılık: {summ['consistency_with_xai2']}")
        print(f"  gerçekçilik: {summ['realism']['d_nn_cf_median']} / gerçek {summ['realism']['d_nn_real_known_median']:.3f}"
              f"  ({time.perf_counter() - t0:.0f} s, {f.calls:,} model satırı)")
        # 7) eş kontrolü: bilinmeyen grupların tamamı için en yakın bilinen eğitim satırı ve etiketi
        nn_full = NearestNeighbors(n_neighbors=1).fit(Xs)
        cat_fit = category[tr_fit]
        twins = {}
        for g, m in (("alarm_unknown", alarm_te & is_unk), ("guard_unknown", flag & is_unk), ("silent_unknown", silent)):
            gi = np.where(m)[0]
            if len(gi) == 0:
                twins[g] = {"n": 0}
                continue
            dist, nb = nn_full.kneighbors(Xte_np[gi] / scale)
            dist, lab = dist[:, 0], cat_fit[nb[:, 0]]
            exact = dist < 1e-9
            share = lambda labs: {str(k): round(float(v) / len(labs), 4)
                                  for k, v in zip(*np.unique(labs, return_counts=True))} if len(labs) else {}
            twins[g] = {"n": int(len(gi)), "d_median": float(np.median(dist)), "exact_twin_share": round(float(exact.mean()), 4),
                        "nearest_label_share": share(lab), "exact_twin_label_share": share(lab[exact])}
        print("  eş kontrolü: " + "; ".join(f"{g}: n={t['n']}, birebir eş %{100 * t.get('exact_twin_share', 0):.0f}, "
                                            f"en yakın etiket {t.get('nearest_label_share', {})}" for g, t in twins.items()))
        out_fam[fam] = {"categories": fam_cats, "decision_source": source, "model": model_src, "twins": twins,
                        "check_vs_dss_guard": check, "tau_alarm": tau_alarm, "tau_guard": tau_guard,
                        "score_iqr": iqr, "summary": summ, "rows": rows, "model_calls": f.calls,
                        "seconds": round(time.perf_counter() - t0, 1)}

    out = {"dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop,
           "status": "Analiz planı sonuçlardan önce yazıldı (README, Kaçışların Karşı-Olgusal Açıklaması / XAI-3); eşik yok",
           "settings": {k: v for k, v in vars(args).items()}, "port_columns": ports,
           "grid_percentiles": QUANTS.tolist(), "families": out_fam,
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    path = os.path.join(args.out_dir, f"xai_escape_counterfactual_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plot(out, os.path.join(args.out_dir, f"fig_xai_escape_counterfactual_{args.dataset}_{args.condition}.png"))
    print(f"\nÇıktı: {path}  ({out['runtime_seconds']:.0f} s)")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [(k, v["summary"]) for k, v in out["families"].items() if "summary" in v]
    if not fams:
        return
    fig, axes = plt.subplots(2, len(fams), figsize=(4.2 * len(fams), 6.2), squeeze=False)
    for c, (fam, s) in enumerate(fams):
        ax = axes[0, c]
        cc = s["caught_curve"]
        ax.plot(cc["ks"], np.array(cc["greedy"]) * 100, marker="o", color="#2a78d6", label="karşı-olgu (açgözlü)")
        ax.plot(cc["ks"], np.array(cc["random"]) * 100, marker="o", color="#9aa5b1", label="rastgele")
        ce = s["energy_only"]["caught_curve"]
        ax.plot(ce["ks"], np.array(ce["greedy"]) * 100, marker="s", ls="--", color="#d6762a",
                label="karşı-olgu, yalnız enerji")
        ax.plot(ce["ks"], np.array(ce["random"]) * 100, marker="s", ls="--", color="#e3b48a",
                label="rastgele, yalnız enerji")
        ax.set_title(f"{fam}: sessiz kaçanlar (n={s['n_explained']})", fontsize=9)
        ax.set_xlabel("değiştirilen özellik sayısı k", fontsize=8)
        ax.set_ylabel("yakalanan (%)", fontsize=8)
        ax.set_xticks(cc["ks"])
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax2 = axes[1, c]
        sh_any = {t["feature"]: t["share_of_changes"] * 100 for t in s["top_changed_features"]}
        sh_e = {t["feature"]: t["share_of_changes"] * 100 for t in s["energy_only"]["top_changed_features"]}
        feats = sorted(set(sh_any) | set(sh_e), key=lambda k: max(sh_any.get(k, 0), sh_e.get(k, 0)))[-8:]
        yy = np.arange(len(feats))
        ax2.barh(yy + 0.2, [sh_any.get(k, 0) for k in feats], 0.4, color="#2a78d6", label="alarm veya enerji")
        ax2.barh(yy - 0.2, [sh_e.get(k, 0) for k in feats], 0.4, color="#d6762a", label="yalnız enerji")
        ax2.set_yticks(yy)
        ax2.set_yticklabels(feats)
        ax2.set_xlabel("değişikliklerin payı (%)", fontsize=8)
        ax2.set_title("karşı-olguda değişen özellikler", fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(labelsize=7)
        if c == 0:
            ax2.legend(frameon=False, fontsize=7)
        if cc["greedy"] == ce["greedy"]:
            ax.text(0.98, 0.9, "iki karşı-olgu eğrisi üst üste", transform=ax.transAxes, ha="right", fontsize=7,
                    color="#555")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.suptitle(f"{out['dataset_id']}: kaçışların karşı-olgusal açıklaması", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
