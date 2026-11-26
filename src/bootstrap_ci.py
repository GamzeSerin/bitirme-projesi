"""Aile başına örnek sayıları ve bootstrap %95 güven aralıkları (#26).

Analiz planı (sonuçlardan önce, 26.11.2026): data/dnn_edgeiiot/README.md, "Aile başına örnek sayıları ve bootstrap %95
güven aralıkları (#26)".

Model eğitimi yok, veri seti gerekmez. Girdi: two_signal.py satır dosyaları (rows_<AİLE>_<koşul>.npz) ve sonuç JSON'u,
tohum klasörü başına.

Kurallar (plandaki gibi, sabit):
- Kararların yeniden üretimi: kontrol eşiği kaydedilmemiş. Her aile/tohum/q için normal trafikte JSON'daki inceleme
  sayısı k (benign_review x normal satır) ve bilinmeyenlerde sessiz kaçış sayısı m (silent_benign x bilinmeyen satır)
  birlikte tutturulur: kontrole ulaşan normal satırlardan enerjisi en yüksek k tanesi işaretlenir, kontrole ulaşan
  bilinmeyenlerden enerjisi en düşük m tanesi sessiz geçer. Bunu sağlayan tek bir eşik olduğu kontrol edilir (yoksa durur).
- Bootstrap: bilinmeyen ve normal satırlar ayrı ayrı, yerine koyarak, grup büyüklüğü sabit; B = 2000, tohum 0;
  yüzdelik aralık. Satırlar ikili göstergelerden oluştuğu için çekiliş, satır TİPLERİ üzerinde çok terimli (multinomial)
  sayımla yapılır; bu, satır satır yeniden çekmeyle aynı dağılımdır, yalnız hızlıdır.
- 5 aile ortalaması: her tekrarda aileler bağımsız çekilir, aile değerlerinin ortalaması alınır.
Ana ölçüt (plandan): q = %5'te 5 aile ortalamasında (kontrolsüz - enerji) farkının %95 alt ucu > 0, her tohumda.

Çıktı: <out_dir>/bootstrap_ci_<koşul>.json (+ "runtime").
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.contract import environment_info  # noqa: E402

FAMILIES = ["DDOS", "RECON", "WEB", "BRUTEFORCE", "MALWARE"]
BENIGN, GUARD, QS = "BENIGN", "tabm_energy", (0.01, 0.05, 0.1)


def decisions(z, energy_rows):
    """Satır başına göstergeler. Bilinmeyen: [kontrolsüz sessiz, q başına enerjiyle sessiz]; normal: [alarm, q başına kontrol, q başına toplam]."""
    cat, unk, pred, alarm = z["category"], z["is_unknown"], z["tabm_pred"], z["alarm"]
    e = z["score_tabm_energy"].astype(np.float64)
    ben = (cat == BENIGN) & ~unk
    tg = ~alarm & (pred == BENIGN)
    U = [tg[unk]]
    Bn = [alarm[ben]]
    for q in QS:
        r = energy_rows[q]
        k = int(round(r["benign_review"] * ben.sum()))
        m = int(round(r["silent_benign"] * unk.sum()))
        eb, eu = e[ben & tg], e[unk & tg]
        hi = np.sort(eb)[::-1][k - 1] if k > 0 else np.inf          # işaretlenen en düşük normal enerjisi
        lo = np.sort(eb)[::-1][k] if k < len(eb) else -np.inf      # işaretlenmeyen en yüksek normal enerjisi
        su = np.sort(eu)
        t_lo = su[m - 1] if m > 0 else -np.inf                       # sessiz geçen en yüksek bilinmeyen
        t_hi = su[m] if m < len(su) else np.inf                      # işaretlenen en düşük bilinmeyen
        tau_lo, tau_hi = max(lo, t_lo), min(hi, t_hi)                # eşik: tau_lo <= tau < tau_hi
        if not tau_lo < tau_hi:
            raise ValueError(f"eşik bulunamadı (q={q}): [{tau_lo}, {tau_hi})")
        tau = tau_lo
        flag = tg & (e > tau)
        assert (flag & ben).sum() == k and (tg & ~flag & unk).sum() == m
        U.append((tg & ~flag)[unk])
        Bn.append(flag[ben])
    Bn += [Bn[0] | f for f in Bn[1:]]                                  # toplam yük: alarm VEYA kontrol
    return np.column_stack(U).astype(int), np.column_stack(Bn).astype(int)


def type_counts(M):
    types, counts = np.unique(M, axis=0, return_counts=True)
    return types.astype(float), counts


def boot(M, B, rng):
    """M: satır x gösterge (0/1). Dönüş: B x gösterge, her tekrarda sütun ortalaması."""
    T, c = type_counts(M)
    draws = rng.multinomial(c.sum(), c / c.sum(), size=B)            # B x tip
    return draws @ T / c.sum()


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)]


def one_seed(run_dir, condition, B, rng):
    with open(os.path.join(run_dir, f"two_signal_{condition}.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    fam_out, boots = {}, {}
    for fam in FAMILIES:
        z = np.load(os.path.join(run_dir, f"rows_{fam}_{condition}.npz"), allow_pickle=True)
        er = {r["q"]: r for r in res["results"] if r["guard"] == GUARD and r["family"] == fam}
        U, Bn = decisions(z, er)
        bu, bb = boot(U, B, rng), boot(Bn, B, rng)
        boots[fam] = (bu, bb)
        f = {"n_known_test": int((~z["is_unknown"]).sum()), "n_unknown_test": int(len(U)),
             "n_benign_test": int(len(Bn)), "n_silent_alarm_only": int(U[:, 0].sum()),
             "silent_alarm_only": {"est": round(float(U[:, 0].mean()), 5), "ci95": ci(bu[:, 0])}, "q": {}}
        for j, q in enumerate(QS, start=1):
            f["q"][str(q)] = {
                "n_silent_energy": int(U[:, j].sum()),
                "silent_energy": {"est": round(float(U[:, j].mean()), 5), "ci95": ci(bu[:, j])},
                "diff_alarm_only_minus_energy": {"est": round(float((U[:, 0] - U[:, j]).mean()), 5),
                                                 "ci95": ci(bu[:, 0] - bu[:, j])},
                "guard_load": {"est": round(float(Bn[:, j].mean()), 5), "ci95": ci(bb[:, j])},
                "total_benign_load": {"est": round(float(Bn[:, j + len(QS)].mean()), 5), "ci95": ci(bb[:, j + len(QS)])}}
        fam_out[fam] = f
    mean = {"silent_alarm_only": None, "q": {}}
    mu = np.mean([boots[f][0][:, 0] for f in FAMILIES], axis=0)
    mean["silent_alarm_only"] = {"est": round(float(np.mean([fam_out[f]["silent_alarm_only"]["est"] for f in FAMILIES])), 5),
                                 "ci95": ci(mu)}
    for j, q in enumerate(QS, start=1):
        se = np.mean([boots[f][0][:, j] for f in FAMILIES], axis=0)
        gl = np.mean([boots[f][1][:, j] for f in FAMILIES], axis=0)
        est = lambda key: round(float(np.mean([fam_out[f]["q"][str(q)][key]["est"] for f in FAMILIES])), 5)
        mean["q"][str(q)] = {"silent_energy": {"est": est("silent_energy"), "ci95": ci(se)},
                             "diff_alarm_only_minus_energy": {"est": est("diff_alarm_only_minus_energy"), "ci95": ci(mu - se)},
                             "guard_load": {"est": est("guard_load"), "ci95": ci(gl)},
                             "ci95_half_width_silent_energy": round(float((np.percentile(se, 97.5) - np.percentile(se, 2.5)) / 2), 5)}
    return {"families": fam_out, "mean_over_families": mean}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", default=["outputs/dnn_edgeiiot/two_signal",
                                                  "outputs/dnn_edgeiiot/two_signal/s1",
                                                  "outputs/dnn_edgeiiot/two_signal/s2"])
    ap.add_argument("--condition", default="C_seq_ack_yok")
    ap.add_argument("--out-dir", default="outputs/dnn_edgeiiot/bootstrap_ci")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = {}
    for d in args.runs:
        with open(os.path.join(d, f"two_signal_{args.condition}.json"), encoding="utf-8") as fh:
            seed = json.load(fh)["settings"].get("model_seed") or 42  # tohum 42 koşusunda alan boş
        seeds[str(seed)] = one_seed(d, args.condition, args.B, rng)
    # tohum sapması (aynı ölçümlerin tohumlar arası sd'si), bootstrap yarı genişliğiyle yan yana
    seed_sd = {}
    for q in QS:
        vals = [s["mean_over_families"]["q"][str(q)]["silent_energy"]["est"] for s in seeds.values()]
        hw = [s["mean_over_families"]["q"][str(q)]["ci95_half_width_silent_energy"] for s in seeds.values()]
        seed_sd[str(q)] = {"silent_energy_seed_sd": round(float(np.std(vals, ddof=1)), 5),
                           "silent_energy_boot_half_width_mean": round(float(np.mean(hw)), 5)}
    lows = [s["mean_over_families"]["q"]["0.05"]["diff_alarm_only_minus_energy"]["ci95"][0] for s in seeds.values()]
    verdict = ("sağlandı (her tohumda farkın %95 alt ucu > 0)" if all(x > 0 for x in lows)
               else "en az bir tohumda aralık 0'ı içeriyor")
    out = {"dataset_id": "dnn_edgeiiot", "condition": args.condition, "guard": GUARD,
           "plan": "data/dnn_edgeiiot/README.md, '#26' (26.11.2026, sonuçlardan önce)",
           "seeds": seeds, "seed_vs_bootstrap": seed_sd, "criterion_q": 0.05, "criterion_diff_ci_low": lows,
           "verdict": verdict, "settings": dict(vars(args))}
    out["runtime"] = environment_info()  # yazılım sürümleri ve donanım (common/contract.py)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"bootstrap_ci_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    for s, r in seeds.items():
        m = r["mean_over_families"]
        print(f"\n=== tohum {s}: kontrolsüz kaçış {m['silent_alarm_only']}")
        for q, v in m["q"].items():
            print(f"  q={q}: enerji {v['silent_energy']}  fark {v['diff_alarm_only_minus_energy']}  yük {v['guard_load']}")
        for fam, f in r["families"].items():
            v = f["q"]["0.05"]
            print(f"  {fam:10s} bilinen {f['n_known_test']:6d} bilinmeyen {f['n_unknown_test']:6d} normal {f['n_benign_test']:5d} "
                  f"kaçan(kontrolsüz) {f['n_silent_alarm_only']:5d} kaçan(q=5) {v['n_silent_energy']:4d}  "
                  f"enerji {v['silent_energy']}  yük {v['guard_load']}")
    print("\ntohum sd vs bootstrap yarı genişlik:", seed_sd)
    print(f"Ana ölçüt: farkın alt uçları {lows} -> {verdict}\n{path}")


if __name__ == "__main__":
    main()
