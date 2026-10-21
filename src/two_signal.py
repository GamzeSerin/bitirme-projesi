"""İki sinyalli karar kuralı: alarm + BENIGN güvenlik kontrolü (ekip ortak aracı, --dataset).

Soru: CP-TabM, görülmemiş bir saldırının sessizce "normal trafik" diye geçirilmesini (sessiz BENIGN
kaçışı) analist yükünü makul tutarak önleyebilir mi? Ve bunu sade TabM'in kendi skorlarıyla yapılan
aynı türden bir kontrolden daha iyi yapar mı?

Karar akışı (her test satırı için):
  1. Sınıflandırıcı: sade TabM (en iyi bilinen ve en kararlı model). Tahmin ve alarm skoru ondan gelir.
  2. Alarm: TabM üye anlaşmazlığı > tau1 ise "bilinmeyen" (tau1 = bilinen kalibrasyonun %95'i).
  3. Alarm yoksa ve tahmin BENIGN ise GÜVENLİK KONTROLÜ: kontrol skoru g(x) > tau_g ise trafik
     geçirilmez, analiste gider (DSS'te ALLOW yerine ANALYST_REVIEW).
     tau_g, kalibrasyondaki gerçek BENIGN satırlarının yalnızca q kadarını işaretleyecek şekilde seçilir
     (q = analist yükü bütçesi; test ve bilinmeyen aile kullanılmaz).

Karşılaştırılan kontrol skorları (aynı q bütçesinde):
  cp_benign_z     CP-TabM: BENIGN uzmanlarının gerçek BENIGN'e göre ne kadar az emin olduğu (ayar aktarımı,
                  tanımlayıcı karşılaştırma)
  tabm_disagree   TabM üye anlaşmazlığı (alarmla aynı skor, BENIGN için daha sıkı eşik)
  tabm_msp        TabM 1 - en yüksek olasılık
  tabm_energy     TabM enerjisi  <- ANA KONTROL SKORU (ekip kararı 21.10.2026, sonuçlardan önce)
  knn             modelden bağımsız uzaklık

Ölçütler (aile başına, sonra ortalama):
  silent_benign       bilinmeyenlerin alarm ALMAYIP, BENIGN denip kontrolden de GEÇTİĞİ oran (en tehlikeli)
  benign_review       gerçek BENIGN test trafiğinin analiste gönderilen oranı (maliyet)
  unknown_caught      bilinmeyenlerin alarm ya da kontrolle durdurulan oranı

Satır bazında skorlar da kaydedilir (XAI-2 ve sonraki analizler için).

Çalıştırma:
  python src/two_signal.py --dataset dnn_edgeiiot --data-file <csv> --cp-config outputs/dnn_edgeiiot/cp_tune/cp_tune_C_seq_ack_yok.json
  python src/two_signal.py --dataset dataset_23 --data-file <klasör>        # ayar aktarımı: kör oranı 0.0625
      (Edge-IIoTset iç provasında 5 ailenin 4'ünde seçilen değer; hedef veride ayar YAPILMAZ.
       JSON'da her aile için blind_frac_source alanı bunu kaydeder.)
  ... --quick
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from open_set_lofo import KNNScore, model_scores

QS = [0.005, 0.01, 0.02, 0.05, 0.10]
GUARDS = ["cp_benign_z", "tabm_disagree", "tabm_msp", "tabm_energy", "knn"]
PRIMARY_GUARD = "tabm_energy"  # önceden tek seçildi (data/dataset_23/README.md, 21.10.2026)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/two_signal")
    p.add_argument("--condition", default=None, help="Varsayılan: ana koşul")
    p.add_argument("--families", default="")
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=None)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95, help="Alarm eşiği: kabul edilen bilinen oranı")
    p.add_argument("--cp-config", default="", help="cp_tune JSON'u (aile başına kör oranı)")
    p.add_argument("--cp-blind-frac", type=float, default=0.0625)
    p.add_argument("--save-rows", action="store_true", default=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap = 3000
    spec = registry.get(args.dataset)
    args.condition = args.condition or spec["default_condition"]
    mseed = args.seed if args.model_seed is None else args.model_seed
    # Tohumlar birbirinin üzerine yazmasın: bölme tohumundan farklı her model tohumu kendi klasörüne
    # (open_set_lofo ve seed_check ile aynı düzen: .../two_signal/s1, s2)
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}/two_signal"
        if mseed != args.seed:
            args.out_dir = os.path.join(args.out_dir, f"s{mseed}")
    os.makedirs(args.out_dir, exist_ok=True)
    # Dosya adı da tohumu taşır: --out-dir elle verilse bile tohumlar birbirini silmez
    tag = args.condition if mseed == args.seed else f"{args.condition}_seed{mseed}"
    epochs = 3 if args.quick else 60
    t_start = time.perf_counter()

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, drop = registry.apply_condition(X, spec, args.condition)
    benign = spec["benign"]
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    fam_def = spec["families"]
    wanted = [f for f in args.families.split(",") if f] or list(fam_def)
    families = [f for f in wanted if set(fam_def[f]) & cats]
    cp_cfg = {}
    if args.cp_config:
        with open(args.cp_config, encoding="utf-8") as f:
            cp_cfg = {fam: v["chosen"] for fam, v in json.load(f)["families"].items()}
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                      random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, koşul {args.condition}; aileler {families}; "
          f"analist bütçeleri q = {QS}")

    from cp_tabm_classifier import CPTabMClassifier
    from tabm_classifier import TabMClassifier

    rows, details = [], {}
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

        tabm = TabMClassifier(max_epochs=epochs, random_state=mseed, verbose=False).fit(Xfit, yfit)
        bf = cp_cfg.get(fam, {}).get("blind_frac", args.cp_blind_frac)
        cp = CPTabMClassifier(blind_frac=bf, max_epochs=epochs, random_state=mseed, verbose=False).fit(Xfit, yfit)
        knn = KNNScore(seed=args.seed).fit(Xfit)

        P_cal, S_cal = model_scores("TabM", tabm, Xcal)
        P_te, S_te = model_scores("TabM", tabm, Xte)
        pred_cal, pred_te = P_cal.argmax(1), P_te.argmax(1)
        G_cal = {"cp_benign_z": cp.class_energy_z(Xcal, b), "tabm_disagree": S_cal["disagree"],
                 "tabm_msp": S_cal["msp"], "tabm_energy": S_cal["energy"], "knn": knn.score(Xcal)}
        G_te = {"cp_benign_z": cp.class_energy_z(Xte, b), "tabm_disagree": S_te["disagree"],
                "tabm_msp": S_te["msp"], "tabm_energy": S_te["energy"], "knn": knn.score(Xte)}

        tau1 = float(np.quantile(S_cal["disagree"], args.target))
        alarm = S_te["disagree"] > tau1
        to_guard = (~alarm) & (pred_te == b)               # alarm yok ve "normal" dendi
        base_silent = float((to_guard & is_unk).sum() / is_unk.sum())
        base_caught = float((alarm & is_unk).sum() / is_unk.sum())
        print(f"\n=== {fam} {fam_cats} (kör oranı {bf}) — yalnız alarm: yakalanan {base_caught:.3f}, "
              f"sessiz BENIGN kaçışı {base_silent:.3f} ===")
        details[fam] = {"categories": fam_cats, "blind_frac": bf,
                        "blind_frac_source": "cp_tune (iç prova)" if fam in cp_cfg else "ayar aktarımı (sabit, --cp-blind-frac)", "tau_alarm": tau1,
                        "alarm_only": {"unknown_caught": round(base_caught, 4), "silent_benign": round(base_silent, 4)},
                        "n_unknown": int(is_unk.sum()), "n_benign_test": int(is_ben.sum())}
        # kontrol eşiği: kalibrasyonda alarm almamış ve BENIGN denmiş GERÇEK BENIGN satırlarına göre
        ref = (~(S_cal["disagree"] > tau1)) & (pred_cal == b) & cal_ben
        for g in GUARDS:
            for q in QS:
                tau_g = float(np.quantile(G_cal[g][ref], 1 - q))
                flag = to_guard & (G_te[g] > tau_g)
                silent = float((to_guard & ~flag & is_unk).sum() / is_unk.sum())
                review = float((flag & is_ben).sum() / is_ben.sum())
                caught = float(((alarm | flag) & is_unk).sum() / is_unk.sum())
                rows.append({"family": fam, "guard": g, "q": q, "silent_benign": round(silent, 5),
                             "benign_review": round(review, 5), "unknown_caught": round(caught, 4),
                             "silent_benign_alarm_only": round(base_silent, 5)})
            r = [x for x in rows if x["family"] == fam and x["guard"] == g]
            print(f"  {g:14s} " + "  ".join(f"q={x['q']:.3f}: kaçış {x['silent_benign']:.4f} / "
                                            f"inceleme {x['benign_review']:.4f}" for x in r if x["q"] in (0.01, 0.05)))
        if args.save_rows:
            np.savez_compressed(os.path.join(args.out_dir, f"rows_{fam}_{tag}.npz"),
                                test_index=idx_te, category=category[idx_te], is_unknown=is_unk,
                                tabm_pred=np.array(known)[pred_te], alarm=alarm, tau_alarm=tau1,
                                **{f"score_{k}": v for k, v in G_te.items()})
        print(f"  ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    mean = df.groupby(["guard", "q"])[["silent_benign", "benign_review", "unknown_caught",
                                       "silent_benign_alarm_only"]].mean().round(5).reset_index()
    out = {"dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop,
           "classifier": "TabM (alarm: üye anlaşmazlığı, %95 bilinen kalibrasyon eşiği)",
           "guard_threshold": "kalibrasyonda alarm almamış ve BENIGN denmiş gerçek BENIGN satırlarının (1-q) yüzdeliği",
           "settings": {**{k: v for k, v in vars(args).items() if k != "data_file"}, "model_seed": mseed},
           "primary_guard": PRIMARY_GUARD,
           "families": details, "results": rows, "mean_over_families": mean.to_dict(orient="records"),
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out_dir, f"two_signal_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot(mean, df, spec["name"], args.condition,
         os.path.join(args.out_dir, f"fig_two_signal_{args.dataset}_{tag}.png"), len(families))
    print("\nAilelerin ortalaması (q = analist bütçesi):")
    print(mean.pivot_table(index="guard", columns="q", values=["silent_benign", "benign_review"]).round(4).to_string())
    print(f"\nÇıktılar: {args.out_dir}  ({time.perf_counter() - t_start:.0f} s)")


def plot(mean, df, name, condition, out_path, n_families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"cp_benign_z": "#1f9e6e", "tabm_disagree": "#2a78d6", "tabm_msp": "#9aa5b1",
              "tabm_energy": "#7b4fd6", "knn": "#d9822b"}
    labels = {"cp_benign_z": "CP-TabM · BENIGN uzmanları", "tabm_disagree": "TabM · anlaşmazlık",
              "tabm_msp": "TabM · MSP", "tabm_energy": "TabM · enerji (ana)", "knn": "kNN"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    base = mean["silent_benign_alarm_only"].iloc[0]
    for g in GUARDS:
        m = mean[mean.guard == g].sort_values("q")
        ax.plot(m["benign_review"] * 100, m["silent_benign"] * 100, marker="o", color=colors[g], label=labels[g],
                lw=2.2 if g == PRIMARY_GUARD else 1.4)
    ax.axhline(base * 100, color="#14202b", ls="--", lw=0.9)
    ax.text(ax.get_xlim()[1] * 0.98, base * 100, "  kontrol yok (yalnız alarm)", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Analiste gönderilen gerçek BENIGN trafik (%)")
    ax.set_ylabel("Sessizce BENIGN'e kaçan görülmemiş saldırı (%)")
    ax.set_title(f"{name}, koşul {condition}: BENIGN güvenlik kontrolü ({n_families} ailenin ortalaması)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
