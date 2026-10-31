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

Derin öğrenme karşılaştırması (--classifier; analiz planı sonuçlardan önce README'de, 26.10.2026):
  TabM       k = 32 üyeli topluluk; alarm = üye anlaşmazlığı (varsayılan, yukarıdaki akış)
  MLP        aynı mimari ve eğitim, k = 1 (tek ağ); anlaşmazlık tanımsız, alarm = MSP (1 - en yüksek olasılık)
  MCDropout  aynı tek ağ (aynı tohumla aynı ağırlıklar); tahminde dropout açık, T = --mc-passes geçiş;
             alarm = geçişler arası anlaşmazlık, enerji = geçişlerin ortalama logitinden
  MLP/MCDropout'ta CP-TabM kontrolü yoktur; kontroller: anlaşmazlık (yalnız MCDropout), MSP, enerji (ana), kNN.
  Ek ölçütler (her sınıflandırıcıda): alarm skoru ve enerji için AUROC (bilinmeyen / bilinen test), bilinen test
  sınıflarında macro-F1, eğitim süresi.

Satır bazında skorlar da kaydedilir (Bilinmeyen Kararının Açıklanması / XAI-2 ve sonraki analizler için).

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
CLASSIFIERS = {  # ad: (önek, alarm skoru, kontroller)
    "TabM": ("tabm", "disagree", GUARDS),
    "MLP": ("mlp", "msp", ["mlp_msp", "mlp_energy", "knn"]),
    "MCDropout": ("mcd", "disagree", ["mcd_disagree", "mcd_msp", "mcd_energy", "knn"]),
}


def mc_dropout_logits(model, Xt, passes, seed, batch=8192):
    """Eğitilmiş tek ağın (k = 1) dropout açık T geçişi: (n, T, C) logit. Diğer katmanlar değerlendirme modunda."""
    import torch
    net = model.model_
    drops = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    if not drops or all(m.p == 0 for m in drops):
        raise SystemExit("MCDropout: ağda dropout katmanı yok")
    torch.manual_seed(seed)
    net.eval()
    for m in drops:
        m.train()
    dev = next(net.parameters()).device
    out = []
    try:
        with torch.inference_mode():
            for i in range(0, len(Xt), batch):
                xb = torch.as_tensor(Xt[i:i + batch], device=dev)
                out.append(torch.cat([net(xb).float() for _ in range(passes)], dim=1).cpu())
    finally:
        net.eval()
    return torch.cat(out).numpy()


def classifier_scores(name, model, X, passes=32, seed=42):
    """(P, skorlar) TabM / MLP / MCDropout için aynı tanımlarla (open_set_lofo.scores_from_members)."""
    if name == "MCDropout":
        from open_set_lofo import scores_from_members
        logits = mc_dropout_logits(model, model._prep(X), passes, seed)
        z = logits - logits.max(-1, keepdims=True)
        return scores_from_members(np.exp(z) / np.exp(z).sum(-1, keepdims=True), logits)
    return model_scores("TabM", model, X)   # MLP: k = 1 (anlaşmazlık 0)


def extra_metrics(is_unk, alarm_score, energy, pred_names, cat_names, known):
    """Alarm skoru ve enerji için AUROC (bilinmeyen = 1), bilinen test satırlarında macro-F1."""
    from sklearn.metrics import f1_score, roc_auc_score
    kn = ~is_unk
    return {"alarm_auroc": round(float(roc_auc_score(is_unk, alarm_score)), 4),
            "energy_auroc": round(float(roc_auc_score(is_unk, energy)), 4),
            "known_macro_f1": round(float(f1_score(cat_names[kn], pred_names[kn], labels=known, average="macro")), 4)}


def temporal_split(idx, fine, test_frac):
    """Her ince etikette idx'leri kayıt sırasına (artan indeks = CSV satır sırası) göre dizer; son test_frac'ı ayırır."""
    a, b = [], []
    for lab in np.unique(fine[idx]):
        ix = np.sort(idx[fine[idx] == lab])
        cut = int(round((1 - test_frac) * len(ix)))
        a.append(ix[:cut]); b.append(ix[cut:])
    return np.concatenate(a), np.concatenate(b)


def recalibration(fam, guards, alarm_key, target, b, tau1, S_cal, G_cal, pred_cal, cal_ben,
                  S_te, G_te, pred_te, is_unk, is_ben, in_w1):
    """Aynı modelin skorlarıyla üç eşik seçimi, hepsi W2'de değerlendirilir (README, 'Zaman bölmesinde yeniden kalibrasyon').
    base: kalibrasyon (eğitimin son %10'u). R1: alarm eşiği aynı, kontrol eşiği W1'in gerçek BENIGN satırlarından.
    R2: alarm eşiği W1'in bilinen satırlarından (target yüzdeliği) ve kontrol eşiği W1'den. Görülmemiş satırlar W1'de kullanılmaz."""
    w2 = ~in_w1
    u2, n2 = is_unk & w2, is_ben & w2
    kn_w1 = in_w1 & ~is_unk
    tau_r2 = float(np.quantile(S_te[alarm_key][kn_w1], target))
    out = []
    for v, ta in (("base", tau1), ("R1", tau1), ("R2", tau_r2)):
        alarm = S_te[alarm_key] > ta
        to_guard = (~alarm) & (pred_te == b)
        if v == "base":
            ref_scores = {g: G_cal[g][(~(S_cal[alarm_key] > tau1)) & (pred_cal == b) & cal_ben] for g in guards}
        else:
            ref = in_w1 & is_ben & ~alarm & (pred_te == b)
            ref_scores = {g: G_te[g][ref] for g in guards}
        a_ben = float((alarm & n2).sum() / max(n2.sum(), 1))
        for g in guards:
            for q in QS:
                tau_g = float(np.quantile(ref_scores[g], 1 - q))
                flag = to_guard & (G_te[g] > tau_g)
                rev = float((flag & n2).sum() / max(n2.sum(), 1))
                out.append({"family": fam, "variant": v, "guard": g, "q": q, "tau_alarm": ta, "tau_guard": tau_g,
                            "n_ref": int(len(ref_scores[g])), "n_unknown_w2": int(u2.sum()), "n_benign_w2": int(n2.sum()),
                            "silent_benign": round(float((to_guard & ~flag & u2).sum() / max(u2.sum(), 1)), 5),
                            "guard_review": round(rev, 5), "alarm_benign": round(a_ben, 5),
                            "total_benign_load": round(float(((alarm | flag) & n2).sum() / max(n2.sum(), 1)), 5),
                            "unknown_caught": round(float(((alarm | flag) & u2).sum() / max(u2.sum(), 1)), 4)})
    return out


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
    p.add_argument("--classifier", choices=list(CLASSIFIERS), default="TabM")
    p.add_argument("--mc-passes", type=int, default=32, help="MCDropout geçiş sayısı (TabM'in k'sı ile aynı)")
    p.add_argument("--split", choices=["random", "time"], default="random",
                   help="random: katmanlı rastgele 80/20 (ana deney). time: her ince etiketin kayıt sırasına göre ilk %%80 "
                        "eğitim, son %%20 test; kalibrasyon eğitimin en son %%10'u (README, zaman bölmesinde iki sinyal)")
    p.add_argument("--skip-cp", action="store_true", help="CP-TabM eğitme (kontrol listesinden cp_benign_z çıkar)")
    p.add_argument("--recal-frac", type=float, default=0.0,
                   help="Yalnız --split time: test penceresinin her ince etikette ilk bu kadarı yeniden kalibrasyon penceresi "
                        "(W1), kalanı değerlendirme (W2); eşikler W1'in bilinen satırlarıyla yeniden seçilir, model aynı "
                        "(README, 'Zaman bölmesinde yeniden kalibrasyon')")
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
    clf = args.classifier
    prefix, alarm_key, guards = CLASSIFIERS[clf]
    if args.skip_cp:
        guards = [g for g in guards if g != "cp_benign_z"]
    primary = f"{prefix}_energy"
    base_tag = args.condition if clf == "TabM" else f"{args.condition}_{clf.lower()}"
    if args.split == "time":
        base_tag += "_time"
    if args.recal_frac > 0:
        if args.split != "time":
            raise SystemExit("--recal-frac yalnız --split time ile")
        base_tag += "_recal"
    tag = base_tag if mseed == args.seed else f"{base_tag}_seed{mseed}"
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
    if args.split == "time":
        # yükleyici satırların CSV (kayıt) sırasını korur (örneklem np.sort ile); her ince etikette sıraya göre kesilir
        idx_tr, idx_te = temporal_split(np.arange(len(category)), fine, 0.2)
    else:
        idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine,
                                          random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, koşul {args.condition}; aileler {families}; "
          f"analist bütçeleri q = {QS}")

    from cp_tabm_classifier import CPTabMClassifier
    from tabm_classifier import TabMClassifier

    rows, details, recal_rows = [], {}, []
    if args.recal_frac > 0:
        # test penceresi her ince etikette kayıt sırasıyla ikiye: ilk recal_frac W1 (yeniden kalibrasyon), kalan W2 (değerlendirme)
        w1_idx, _ = temporal_split(idx_te, fine, 1 - args.recal_frac)
        in_w1 = np.isin(idx_te, w1_idx)
    for fam in families:
        t0 = time.perf_counter()
        fam_cats = fam_def[fam]
        known = [c for c in present if c not in fam_cats]
        b = known.index(benign)
        tr = idx_tr[~np.isin(category[idx_tr], fam_cats)]
        if args.split == "time":
            tr_fit, tr_cal = temporal_split(tr, fine, args.cal_size)
        else:
            tr_fit, tr_cal = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr], random_state=args.seed)
        yfit = np.array([known.index(c) for c in category[tr_fit]])
        Xfit, Xcal, Xte = X.iloc[tr_fit], X.iloc[tr_cal], X.iloc[idx_te]
        is_unk = np.isin(category[idx_te], fam_cats)
        is_ben = category[idx_te] == benign
        cal_ben = category[tr_cal] == benign

        t_fit = time.perf_counter()
        tabm = TabMClassifier(k=32 if clf == "TabM" else 1, max_epochs=epochs, random_state=mseed,
                              verbose=False).fit(Xfit, yfit)
        t_fit = time.perf_counter() - t_fit
        bf = cp_cfg.get(fam, {}).get("blind_frac", args.cp_blind_frac)
        cp = (CPTabMClassifier(blind_frac=bf, max_epochs=epochs, random_state=mseed, verbose=False).fit(Xfit, yfit)
              if clf == "TabM" and not args.skip_cp else None)
        knn = KNNScore(seed=args.seed).fit(Xfit)

        P_cal, S_cal = classifier_scores(clf, tabm, Xcal, args.mc_passes, mseed)
        P_te, S_te = classifier_scores(clf, tabm, Xte, args.mc_passes, mseed)
        pred_cal, pred_te = P_cal.argmax(1), P_te.argmax(1)
        if clf == "TabM":
            G_cal = {"cp_benign_z": cp.class_energy_z(Xcal, b) if cp else None, "tabm_disagree": S_cal["disagree"],
                     "tabm_msp": S_cal["msp"], "tabm_energy": S_cal["energy"], "knn": knn.score(Xcal)}
            G_te = {"cp_benign_z": cp.class_energy_z(Xte, b) if cp else None, "tabm_disagree": S_te["disagree"],
                    "tabm_msp": S_te["msp"], "tabm_energy": S_te["energy"], "knn": knn.score(Xte)}
        else:
            G_cal = {g: (knn.score(Xcal) if g == "knn" else S_cal[g.split("_", 1)[1]]) for g in guards}
            G_te = {g: (knn.score(Xte) if g == "knn" else S_te[g.split("_", 1)[1]]) for g in guards}
        tau1 = float(np.quantile(S_cal[alarm_key], args.target))
        alarm = S_te[alarm_key] > tau1
        to_guard = (~alarm) & (pred_te == b)               # alarm yok ve "normal" dendi
        base_silent = float((to_guard & is_unk).sum() / is_unk.sum())
        base_caught = float((alarm & is_unk).sum() / is_unk.sum())
        print(f"\n=== {fam} {fam_cats} (kör oranı {bf}) — yalnız alarm: yakalanan {base_caught:.3f}, "
              f"sessiz BENIGN kaçışı {base_silent:.3f} ===")
        details[fam] = {"categories": fam_cats, "blind_frac": bf,
                        "blind_frac_source": "cp_tune (iç prova)" if fam in cp_cfg else "ayar aktarımı (sabit, --cp-blind-frac)", "tau_alarm": tau1,
                        "alarm_only": {"unknown_caught": round(base_caught, 4), "silent_benign": round(base_silent, 4)},
                        "n_unknown": int(is_unk.sum()), "n_benign_test": int(is_ben.sum()),
                        "classifier": clf, "alarm_score": alarm_key, "train_seconds": round(t_fit, 1),
                        **extra_metrics(is_unk, S_te[alarm_key], S_te["energy"], np.array(known)[pred_te],
                                        category[idx_te], known)}
        if clf != "TabM":
            details[fam].pop("blind_frac"); details[fam].pop("blind_frac_source")
        print(f"  {clf}: alarm AUROC {details[fam]['alarm_auroc']}, enerji AUROC {details[fam]['energy_auroc']}, "
              f"bilinen macro-F1 {details[fam]['known_macro_f1']}, eğitim {t_fit:.0f} s")
        # kontrol eşiği: kalibrasyonda alarm almamış ve BENIGN denmiş GERÇEK BENIGN satırlarına göre
        ref = (~(S_cal[alarm_key] > tau1)) & (pred_cal == b) & cal_ben
        for g in guards:
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
        if args.recal_frac > 0:
            recal_rows += recalibration(fam, guards, alarm_key, args.target, b, tau1, S_cal, G_cal, pred_cal, cal_ben,
                                        S_te, G_te, pred_te, is_unk, is_ben, in_w1)
        if args.save_rows:
            np.savez_compressed(os.path.join(args.out_dir, f"rows_{fam}_{tag}.npz"),
                                test_index=idx_te, category=category[idx_te], is_unknown=is_unk,
                                tabm_pred=np.array(known)[pred_te], alarm=alarm, tau_alarm=tau1,
                                **{f"score_{k}": v for k, v in G_te.items() if v is not None})
        print(f"  ({time.perf_counter() - t0:.0f} s)")

    df = pd.DataFrame(rows)
    mean = df.groupby(["guard", "q"])[["silent_benign", "benign_review", "unknown_caught",
                                       "silent_benign_alarm_only"]].mean().round(5).reset_index()
    out = {"dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop,
           "split": ("zaman: her ince etikette kayıt sırasına göre ilk %80 eğitim / son %20 test; kalibrasyon eğitimin son %10'u"
                     if args.split == "time" else "rastgele: ince etikete göre katmanlı 80/20, kalibrasyon %10"),
           "classifier": ("TabM (alarm: üye anlaşmazlığı, %95 bilinen kalibrasyon eşiği)" if clf == "TabM" else
                          f"{clf} (k = 1; alarm: {alarm_key}, %95 bilinen kalibrasyon eşiği)"),
           "guard_threshold": "kalibrasyonda alarm almamış ve BENIGN denmiş gerçek BENIGN satırlarının (1-q) yüzdeliği",
           "settings": {**{k: v for k, v in vars(args).items() if k != "data_file"}, "model_seed": mseed},
           "primary_guard": primary,
           "families": details, "results": rows, "mean_over_families": mean.to_dict(orient="records"),
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    if recal_rows:
        rdf = pd.DataFrame(recal_rows)
        rmean = rdf.groupby(["variant", "guard", "q"])[["silent_benign", "guard_review", "alarm_benign", "total_benign_load",
                                                        "unknown_caught"]].mean().round(5).reset_index()
        out["recalibration"] = {
            "definition": (f"test penceresi her ince etikette kayıt sırasıyla: ilk %{args.recal_frac * 100:.0f} W1 (yeniden "
                           "kalibrasyon), kalan W2 (değerlendirme; bütün sayılar W2'de). base: eğitimin son %10'undan seçilen "
                           "eşikler; R1: alarm eşiği aynı, kontrol eşiği W1'in gerçek BENIGN satırlarından; R2: alarm eşiği W1'in "
                           "bilinen satırlarından (%95), kontrol eşiği W1'den. Görülmemiş aile W1'de kullanılmaz. Model aynı."),
            "recal_frac": args.recal_frac, "results": recal_rows, "mean_over_families": rmean.to_dict(orient="records")}
        prim = rmean[rmean["guard"] == primary]
        print(f"\nYeniden kalibrasyon (W2'de, {primary}; aile ortalaması):")
        print(prim.pivot_table(index="variant", columns="q", values=["silent_benign", "guard_review", "total_benign_load"])
              .round(4).to_string())
    with open(os.path.join(args.out_dir, f"two_signal_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot(mean, df, spec["name"], args.condition,
         os.path.join(args.out_dir, f"fig_two_signal_{args.dataset}_{tag}.png"), len(families), guards, primary, clf)
    print("\nAilelerin ortalaması (q = analist bütçesi):")
    print(mean.pivot_table(index="guard", columns="q", values=["silent_benign", "benign_review"]).round(4).to_string())
    print(f"\nÇıktılar: {args.out_dir}  ({time.perf_counter() - t_start:.0f} s)")


def plot(mean, df, name, condition, out_path, n_families, guards=GUARDS, primary=PRIMARY_GUARD, clf="TabM"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"cp_benign_z": "#1f9e6e", "tabm_disagree": "#2a78d6", "tabm_msp": "#9aa5b1",
              "tabm_energy": "#7b4fd6", "knn": "#d9822b"}
    labels = {"cp_benign_z": "CP-TabM · BENIGN uzmanları", "tabm_disagree": "TabM · anlaşmazlık",
              "tabm_msp": "TabM · MSP", "tabm_energy": "TabM · enerji (ana)", "knn": "kNN"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    base = mean["silent_benign_alarm_only"].iloc[0]
    for g in guards:
        key = g if g in colors else "tabm_" + g.split("_", 1)[1]
        lab = labels[key] if g in colors else f"{clf} · " + labels[key].split(" · ", 1)[1]
        m = mean[mean.guard == g].sort_values("q")
        ax.plot(m["benign_review"] * 100, m["silent_benign"] * 100, marker="o", color=colors[key], label=lab,
                lw=2.2 if g == primary else 1.4)
    ax.axhline(base * 100, color="#14202b", ls="--", lw=0.9)
    ax.text(ax.get_xlim()[1] * 0.98, base * 100, "  kontrol yok (yalnız alarm)", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("Analiste gönderilen gerçek BENIGN trafik (%)")
    ax.set_ylabel("Sessizce BENIGN'e kaçan görülmemiş saldırı (%)")
    ax.set_title(f"{name}, koşul {condition}, {clf}: BENIGN güvenlik kontrolü ({n_families} ailenin ortalaması)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
