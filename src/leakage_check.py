"""Sızıntı denetimi (#24): test satırının koşul vektörü, o katlamanın eğitim kümesinde birebir var mı?

Analiz planı (sonuçlardan önce, 26.11.2026): data/dnn_edgeiiot/README.md, "Sızıntı denetimi (#24)".

Neden: tekrarlar temizlikte TAM satır üzerinden atılır. Koşul C (tcp.seq, tcp.ack yok) sütun çıkarınca önceden farklı
satırlar birebir aynı olabilir; test vektörü eğitimde de bulunur (paket düzeyinde bölmenin bilinen riski).

İki adım:
  1. --data-file verilirse EŞ BAYRAKLARI hesaplanır (veri gerekir; Kaggle):
     - yükleyici, koşul ve bölme two_signal JSON'undaki ayarlarla aynen kurulur (cap, seed, cal_size; çakışmalar atılmış);
     - kurulan test indeksleri satır dosyalarındaki test_index ile birebir karşılaştırılır; tutmazsa DURUR;
     - her ailede eğitim kümesi = fit + kalibrasyon (two_signal'daki tr; model ya da eşikler ikisini de görür);
     - eşleşme satır özetiyle (pandas hash), eşleşenler sütun sütun doğrulanır;
     - bilinen satırlarda eşlerin kategorisi test satırınınkiyle aynı mı (hepsi aynı / en az biri farklı).
     Bayraklar <out_dir>/twins_<koşul>.npz'ye yazılır (küçük; depoya girer).
  2. Ölçümler (bayraklar + two_signal satır dosyaları, her tohum): eşi olan / olmayan alt kümelerde macro-F1 (bilinen),
     sessiz kaçış (bilinmeyen; kontrolsüz ve enerji q = %5), kontrol yükü (normal). Enerji kararları #26'daki gibi
     yeniden üretilir (bootstrap_ci.decisions).
Ölçütler (plandan): (1) eşi olmayan bilinenlerde macro-F1 tümünden en fazla 0,05 düşük; (2) eşi olmayan bilinmeyenlerde
enerji kontrolü kaçışı kontrolsüze göre en az yarıya indirir; ikisi de 5 aile ortalamasında, her tohumda. Bir alt kümede
100'den az bilinmeyen satır kalan ailede oran yorumlanmaz (JSON'da "small": true).

Çıktı: <out_dir>/leakage_<koşul>.json (+ "runtime").
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import FAMILIES, QS, decisions  # noqa: E402
from common.contract import environment_info  # noqa: E402

BENIGN, MIN_N, Q_MAIN = "BENIGN", 100, 0.05


def compute_twins(args, ref_json, rows_dir):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from common import open_set_registry as registry
    st = ref_json["settings"]
    spec = registry.get(st["dataset"])
    X, fine, category, _ = spec["loader"](args.data_file, st["cap"], st["seed"])
    X, _ = registry.apply_condition(X, spec, st["condition"])
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=st["seed"])
    h = pd.util.hash_pandas_object(X, index=False).to_numpy()
    Xv = X.to_numpy()
    out = {"n_rows": len(category), "n_features": X.shape[1]}
    for fam in FAMILIES:
        z = np.load(os.path.join(rows_dir, f"rows_{fam}_{st['condition']}.npz"), allow_pickle=True)
        if not np.array_equal(z["test_index"], idx_te):
            raise SystemExit(f"{fam}: kurulan test indeksleri satır dosyasındakiyle aynı değil; bölme tutmuyor")
        tr = idx_tr[~np.isin(category[idx_tr], spec["families"][fam])]   # fit + kalibrasyon
        tr_h = pd.Series(category[tr], index=h[tr])
        has = np.isin(h[idx_te], h[tr])
        # eşleşenleri sütun sütun doğrula (özet çakışması olmasın)
        first = pd.Series(tr, index=h[tr]).groupby(level=0).first()
        cand = first.reindex(h[idx_te][has]).to_numpy()
        a, b = Xv[idx_te[has]], Xv[cand]
        same = (a == b) | (pd.isna(a) & pd.isna(b))
        if not same.all():
            raise SystemExit(f"{fam}: özet eşleşen {int((~same.all(1)).sum())} satır sütun sütun farklı")
        cats_by_h = tr_h.groupby(level=0).agg(lambda s: set(s))
        te_cat = category[idx_te]
        twin_same_cat = np.zeros(len(idx_te), dtype=bool)
        hits = np.flatnonzero(has)
        twin_same_cat[hits] = [cats_by_h[hh] == {c} for hh, c in zip(h[idx_te][hits], te_cat[hits])]
        out[fam] = {"has_twin": has, "twin_all_same_category": twin_same_cat}
        print(f"{fam}: eşi olan test satırı {has.mean():.3%} ({int(has.sum())}/{len(has)})")
    return out


def frac(m):
    return round(float(m.mean()), 5) if m.size else None


def measure(run_dir, condition, twins):
    with open(os.path.join(run_dir, f"two_signal_{condition}.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    j = 1 + QS.index(Q_MAIN)
    fams = {}
    for fam in FAMILIES:
        z = np.load(os.path.join(run_dir, f"rows_{fam}_{condition}.npz"), allow_pickle=True)
        er = {r["q"]: r for r in res["results"] if r["guard"] == "tabm_energy" and r["family"] == fam}
        U, Bn = decisions(z, er)
        cat, unk, pred = z["category"], z["is_unknown"], z["tabm_pred"]
        tw = twins[fam]["has_twin"]
        kn, ben = ~unk, (cat == BENIGN) & ~unk
        known = sorted(set(cat[kn]))
        f = {"twin_share": {"known": frac(tw[kn]), "unknown": frac(tw[unk]), "benign": frac(tw[ben])},
             "known_twin_same_category_share": frac(twins[fam]["twin_all_same_category"][kn & tw])}
        for name, sel in (("all", np.ones_like(tw)), ("twin", tw), ("no_twin", ~tw)):
            k_sel, u_sel, b_sel = kn & sel, (unk & sel)[unk], (ben & sel)[ben]
            f[name] = {
                "n_known": int(k_sel.sum()), "n_unknown": int(u_sel.sum()), "n_benign": int(b_sel.sum()),
                "macro_f1": (round(float(f1_score(cat[k_sel], pred[k_sel], labels=known, average="macro",
                                                  zero_division=0)), 4) if k_sel.any() else None),
                "silent_alarm_only": frac(U[u_sel, 0]), "silent_energy_q05": frac(U[u_sel, j]),
                "n_silent_alarm_only": int(U[u_sel, 0].sum()), "n_silent_energy_q05": int(U[u_sel, j].sum()),
                "guard_load_q05": frac(Bn[b_sel, j]), "small": bool(u_sel.sum() < MIN_N)}
        fams[fam] = f
    mean = {}
    for name in ("all", "twin", "no_twin"):
        vals = lambda key: [fams[fm][name][key] for fm in FAMILIES if fams[fm][name][key] is not None]
        mean[name] = {k: round(float(np.mean(vals(k))), 5) if vals(k) else None
                      for k in ("macro_f1", "silent_alarm_only", "silent_energy_q05", "guard_load_q05")}
    return {"families": fams, "mean_over_families": mean}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-file", help="DNN-EdgeIIoT-dataset.csv; verilmezse bayraklar --out-dir'den okunur")
    ap.add_argument("--runs", nargs="+", default=["outputs/dnn_edgeiiot/two_signal",
                                                  "outputs/dnn_edgeiiot/two_signal/s1",
                                                  "outputs/dnn_edgeiiot/two_signal/s2"])
    ap.add_argument("--condition", default="C_seq_ack_yok")
    ap.add_argument("--out-dir", default="outputs/dnn_edgeiiot/leakage")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tpath = os.path.join(args.out_dir, f"twins_{args.condition}.npz")
    with open(os.path.join(args.runs[0], f"two_signal_{args.condition}.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    if args.data_file:
        tw = compute_twins(args, ref, args.runs[0])
        np.savez_compressed(tpath, **{f"{fam}__{k}": v for fam in FAMILIES for k, v in tw[fam].items()},
                            n_rows=tw["n_rows"], n_features=tw["n_features"])
    z = np.load(tpath)
    twins = {fam: {k: z[f"{fam}__{k}"] for k in ("has_twin", "twin_all_same_category")} for fam in FAMILIES}
    seeds = {}
    for d in args.runs:
        with open(os.path.join(d, f"two_signal_{args.condition}.json"), encoding="utf-8") as fh:
            seed = json.load(fh)["settings"].get("model_seed") or 42  # tohum 42 koşusunda alan boş
        seeds[str(seed)] = measure(d, args.condition, twins)
    c1, c2 = [], []
    for s, r in seeds.items():
        m = r["mean_over_families"]
        c1.append(round(m["all"]["macro_f1"] - m["no_twin"]["macro_f1"], 4))
        nt = m["no_twin"]
        c2.append(round(nt["silent_energy_q05"] / nt["silent_alarm_only"], 3) if nt["silent_alarm_only"] else None)
        print(f"\n=== tohum {s}")
        for name in ("all", "twin", "no_twin"):
            print(f"  {name:8s} {m[name]}")
        for fam, f in r["families"].items():
            print(f"  {fam:10s} eş payı {f['twin_share']}  aynı kategori {f['known_twin_same_category_share']}  "
                  f"eşsiz: bilinmeyen {f['no_twin']['n_unknown']}, kaçan {f['no_twin']['n_silent_alarm_only']}"
                  f"->{f['no_twin']['n_silent_energy_q05']}")
    v1 = "sağlandı" if all(d <= 0.05 for d in c1) else "sağlanmadı"
    v2 = "sağlandı" if all(x is not None and x <= 0.5 for x in c2) else "sağlanmadı"
    out = {"dataset_id": "dnn_edgeiiot", "condition": args.condition,
           "plan": "data/dnn_edgeiiot/README.md, '#24' (26.11.2026, sonuçlardan önce)",
           "twins_file": tpath, "n_rows": int(z["n_rows"]), "n_features": int(z["n_features"]),
           "seeds": seeds,
           "criterion1_macro_f1_drop_all_minus_no_twin": c1, "criterion1_verdict": v1,
           "criterion2_no_twin_energy_over_alarm_only": c2, "criterion2_verdict": v2,
           "settings": dict(vars(args))}
    out["runtime"] = environment_info()  # yazılım sürümleri ve donanım (common/contract.py)
    path = os.path.join(args.out_dir, f"leakage_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÖlçüt 1 (macro-F1 düşüşü <= 0,05): {c1} -> {v1}")
    print(f"Ölçüt 2 (eşsizde enerji/kontrolsüz <= 0,5): {c2} -> {v2}\n{path}")


if __name__ == "__main__":
    main()
