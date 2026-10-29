"""Literatürdeki açık küme / dağılım dışı yöntemlerle karşılaştırma (ekip ortak aracı, --dataset).

Soru: Literatürde bilinen yöntemler, TabM'in üye anlaşmazlığından (alarm) ve enerjisinden (BENIGN kontrolü)
daha iyi mi? Analiz planı sonuçlardan önce yazıldı (data/dnn_edgeiiot/README.md, "Literatür karşılaştırması");
eşik yok, aşağıdaki ölçümlerin tamamı kaydedilir.

Protokol open_set_lofo.py ve two_signal.py ile aynı: ekip bölmesi (80/20, seed 42), her ailede LOFO, kalan eğitim
%90 eğitim / %10 kalibrasyon; eşikler yalnız kalibrasyondaki bilinen satırlardan. Her aile ve model tohumu için
TabM bir kez eğitilir; bütün skorlar aynı modelden ya da aynı eğitim verisinden.

Skorlar (büyük = bilinmeyen):
  tabm_disagree   TabM üye anlaşmazlığı (karşılıklı bilgi)          [referans: tezin alarmı]
  tabm_energy     -logsumexp(ort. logit)                              [referans: tezin BENIGN kontrolü]
  tabm_msp        1 - en yüksek olasılık                              [referans: Hendrycks ve Gimpel, 2017]
  knn             5 en yakın komşuya uzaklık                          [referans]
  openmax         OpenMax'in "bilinmeyen" olasılığı (Bendale ve Boult, 2016). Aktivasyon vektörü = üyelerin
                  ortalama logiti; sınıf merkezi = eğitimde doğru sınıflanan satırların ortalaması; uzaklık
                  eucos (öklid/200 + kosinüs); her sınıfta en büyük 20 uzaklığa Weibull; alfa = bilinen sınıf sayısı.
  mahalanobis     Sınıf koşullu Gauss, ortak kovaryans (Ledoit-Wolf), en yakın sınıf merkezine Mahalanobis
                  uzaklığı (Lee vd., 2018; son gizli katman = TabM omurgasının çıktısı, üye ortalaması;
                  girdi pertürbasyonu ve katman topluluğu yok)
  autoencoder     Yeniden üretim hatası (MSE). Bilinen eğitim trafiğiyle eğitilen MLP otokodlayıcı
                  (64-16-64, ReLU, Adam, %10 doğrulamayla erken durdurma, en fazla 50 epoch); girdi TabM'in ön
                  işlenmiş özellikleri. IDS literatüründe yaygın anomali tespiti tabanı.
Yöntemlerin ayarları sabittir (sonuçlara bakılarak seçilmedi).

Ölçümler:
  1. Tek başına "bilinmeyen" skoru olarak: AUROC, kalibrasyon eşiğinde (%95 bilinen kabul) yakalanan bilinmeyen,
     yanlış reddedilen bilinen, BENIGN'e sessiz kaçış (open_set_lofo.evaluate ile aynı).
  2. İki sinyal kuralında BENIGN kontrolü olarak (alarm = TabM anlaşmazlığı, değişmez): q = %1 / 5 / 10'da
     sessiz kaçış, normal trafiğin analiste gitme oranı, yakalanan bilinmeyen (two_signal.py ile aynı tanımlar).
  3. Model tohumları arası ortalama ± standart sapma ve tohum başına TabM referansına göre fark.

Çalıştırma:
  python src/open_set_baselines.py --dataset dnn_edgeiiot --data-file <csv> --seeds 42,1,2
  ... --quick   (küçük örneklem, kısa eğitim: tesisat denemesi)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.special import softmax
from scipy.stats import weibull_min
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES
from open_set_lofo import KNNScore, evaluate, model_scores

QS = [0.01, 0.05, 0.10]
SCORES = ["tabm_disagree", "tabm_energy", "tabm_msp", "knn", "openmax", "mahalanobis", "autoencoder"]
REF_ALARM, REF_GUARD = "tabm_disagree", "tabm_energy"


# ---------------------------------------------------------------- TabM çıktıları
def tabm_outputs(model, X, batch=8192):
    """(ortalama logit (n, C), omurga çıktısı üye ortalaması (n, d)). Omurga çıktısı çıkış katmanının girdisinden
    (ileri kanca) alınır; TabMClassifier dışında model varsa yalnız logit döner."""
    Xt = model._prep(X)
    if not hasattr(model, "model_"):                 # test taslağı vb.: yalnız logit
        return model._logits(Xt).mean(1), None
    import torch
    net = model.model_
    feats, logits = [], []
    hook_out = {}
    h = net.output.register_forward_hook(lambda m, inp, out: hook_out.__setitem__("f", inp[0])) \
        if getattr(net, "output", None) is not None else None
    dev = next(net.parameters()).device
    net.eval()
    try:
        with torch.inference_mode():
            for i in range(0, len(Xt), batch):
                out = net(torch.as_tensor(Xt[i:i + batch], device=dev)).float()
                logits.append(out.mean(1).cpu().numpy())
                if h is not None:
                    feats.append(hook_out["f"].float().mean(1).cpu().numpy())
    finally:
        if h is not None:
            h.remove()
    return np.concatenate(logits), (np.concatenate(feats) if feats else None)


# ---------------------------------------------------------------- OpenMax
class OpenMax:
    def __init__(self, tail=20, alpha=None):
        self.tail, self.alpha = tail, alpha

    @staticmethod
    def _eucos(av, mav):
        cos = 1 - (av @ mav) / (np.linalg.norm(av, axis=1) * np.linalg.norm(mav) + 1e-12)
        return np.linalg.norm(av - mav, axis=1) / 200.0 + cos

    def fit(self, av, y, pred):
        C = av.shape[1]
        self.alpha_ = min(self.alpha or C, C)
        self.mav_, self.wb_ = [], []
        for c in range(C):
            ok = (y == c) & (pred == c)
            if ok.sum() < 2:
                ok = y == c
            mav = av[ok].mean(0)
            d = np.sort(self._eucos(av[ok], mav))[-self.tail:]
            shape, _, scale = weibull_min.fit(np.clip(d, 1e-9, None), floc=0)
            self.mav_.append(mav)
            self.wb_.append((shape, scale))
        return self

    def score(self, av):
        n, C = av.shape
        w = np.zeros((n, C))
        rank = np.argsort(-av, axis=1)
        for c in range(C):
            shape, scale = self.wb_[c]
            wscore = weibull_min.cdf(self._eucos(av, self.mav_[c]), shape, loc=0, scale=scale)
            pos = np.argmax(rank == c, axis=1)                      # sınıfın sırası (0 = en yüksek)
            aw = np.where(pos < self.alpha_, (self.alpha_ - pos) / self.alpha_, 0.0)
            w[:, c] = aw * wscore
        revised = av * (1 - w)
        unk = (av * w).sum(1, keepdims=True)
        return softmax(np.concatenate([revised, unk], axis=1), axis=1)[:, -1]


# ---------------------------------------------------------------- Mahalanobis
class Mahalanobis:
    def fit(self, f, y, max_per_class=20000, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], min(max_per_class, int((y == c).sum())), replace=False)
                              for c in np.unique(y)])
        f, y = f[idx], y[idx]
        self.mu_ = np.stack([f[y == c].mean(0) for c in np.unique(y)])
        f = f.astype(np.float64)
        self.mu_ = self.mu_.astype(np.float64)
        cov = LedoitWolf().fit(f - self.mu_[np.searchsorted(np.unique(y), y)]).covariance_
        # Sayısal sağlamlık: ölü ReLU birimleri kovaryansı neredeyse tekil yapıyor (Cholesky çöküyordu).
        # Özdeğer ayrışımı + taban (en büyük özdeğerin 1e-9'u); d^T Σ^-1 d = |d W|^2, W = V / sqrt(λ).
        lam, V = np.linalg.eigh((cov + cov.T) / 2)
        floor = max(lam.max(), 1e-12) * 1e-9
        self.n_floored_ = int((lam < floor).sum())
        self.W_ = V / np.sqrt(np.maximum(lam, floor))
        return self

    def score(self, f):
        L = self.W_
        fL, muL = f.astype(np.float64) @ L, self.mu_ @ L
        return np.min(np.stack([((fL - m) ** 2).sum(1) for m in muL], axis=1), axis=1)


# ---------------------------------------------------------------- Otokodlayıcı
class AutoEncoder:
    def __init__(self, seed=42, max_iter=50, max_train=100000, quick=False):
        self.seed, self.max_iter, self.max_train, self.quick = seed, max_iter, max_train, quick

    def fit(self, Z):
        rng = np.random.default_rng(self.seed)
        Z = Z[rng.choice(len(Z), min(self.max_train, len(Z)), replace=False)]
        self.m_ = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam",
                               learning_rate_init=1e-3, batch_size=512,
                               max_iter=3 if self.quick else self.max_iter, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=5, random_state=self.seed)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_.fit(Z, Z)
        return self

    def score(self, Z):
        return ((self.m_.predict(Z) - Z) ** 2).mean(1)


# ---------------------------------------------------------------- iki sinyal kuralında kontrol
def guard_eval(alarm_cal, alarm_te, pred_cal, pred_te, b, cal_ben, is_unk, is_ben, g_cal, g_te):
    to_guard = ~alarm_te & (pred_te == b)
    ref = ~alarm_cal & (pred_cal == b) & cal_ben
    out = {"silent_alarm_only": round(float((to_guard & is_unk).sum() / is_unk.sum()), 5)}
    for q in QS:
        tau = float(np.quantile(g_cal[ref], 1 - q)) if ref.any() else np.inf
        flag = to_guard & (g_te > tau)
        out[f"q{q}"] = {"silent": round(float((to_guard & ~flag & is_unk).sum() / is_unk.sum()), 5),
                        "review": round(float((flag & is_ben).sum() / max(is_ben.sum(), 1)), 5),
                        "caught": round(float(((alarm_te | flag) & is_unk).sum() / is_unk.sum()), 4)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--condition", default=None)
    p.add_argument("--families", default="")
    p.add_argument("--seeds", default="42,1,2", help="Model tohumları (veri bölmesi sabit, --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--openmax-tail", type=int, default=20)
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/open_set_baselines")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap = 3000
    spec = registry.get(args.dataset)
    args.condition = args.condition or spec["default_condition"]
    args.out_dir = args.out_dir or f"outputs/{args.dataset}/open_set_baselines"
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    t_start = time.perf_counter()
    from tabm_classifier import TabMClassifier

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, drop = registry.apply_condition(X, spec, args.condition)
    benign = spec["benign"]
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    fam_def = spec["families"]
    families = [f for f in (args.families.split(",") if args.families else fam_def) if f and set(fam_def[f]) & cats]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine, random_state=args.seed)
    print(f"{spec['name']}: {len(category)} satır, koşul {args.condition}; aileler {families}; tohumlar {seeds}")

    res = {}   # seed -> fam -> {"standalone": {score: metrics}, "guard": {score: ...}, ...}
    for mseed in seeds:
        res[mseed] = {}
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
            cal_known = np.isin(category[tr_cal], known)
            cal_ben = category[tr_cal] == benign

            tabm = TabMClassifier(max_epochs=3 if args.quick else 60, random_state=mseed, verbose=False).fit(Xfit, yfit)
            P_fit, _ = model_scores("TabM", tabm, Xfit)
            P_cal, S_cal = model_scores("TabM", tabm, Xcal)
            P_te, S_te = model_scores("TabM", tabm, Xte)
            pred_fit, pred_cal, pred_te = P_fit.argmax(1), P_cal.argmax(1), P_te.argmax(1)
            av_fit, f_fit = tabm_outputs(tabm, Xfit)
            av_cal, f_cal = tabm_outputs(tabm, Xcal)
            av_te, f_te = tabm_outputs(tabm, Xte)
            G_cal = {"tabm_disagree": S_cal["disagree"], "tabm_energy": S_cal["energy"], "tabm_msp": S_cal["msp"]}
            G_te = {"tabm_disagree": S_te["disagree"], "tabm_energy": S_te["energy"], "tabm_msp": S_te["msp"]}
            times = {}
            ts = time.perf_counter()
            knn = KNNScore(seed=args.seed).fit(Xfit)
            G_cal["knn"], G_te["knn"] = knn.score(Xcal), knn.score(Xte)
            times["knn"] = time.perf_counter() - ts
            ts = time.perf_counter()
            om = OpenMax(tail=args.openmax_tail).fit(av_fit, yfit, pred_fit)
            G_cal["openmax"], G_te["openmax"] = om.score(av_cal), om.score(av_te)
            times["openmax"] = time.perf_counter() - ts
            if f_fit is not None:
                ts = time.perf_counter()
                mh = Mahalanobis().fit(f_fit, yfit, seed=mseed)
                G_cal["mahalanobis"], G_te["mahalanobis"] = mh.score(f_cal), mh.score(f_te)
                times["mahalanobis"] = time.perf_counter() - ts
            ts = time.perf_counter()
            ae = AutoEncoder(seed=mseed, quick=args.quick).fit(tabm._prep(Xfit))
            G_cal["autoencoder"], G_te["autoencoder"] = ae.score(tabm._prep(Xcal)), ae.score(tabm._prep(Xte))
            times["autoencoder"] = time.perf_counter() - ts

            pred_names = np.array(known)[pred_te]
            tau_alarm = float(np.quantile(S_cal["disagree"][cal_known], args.target))
            alarm_cal, alarm_te = S_cal["disagree"] > tau_alarm, S_te["disagree"] > tau_alarm
            stand, guard = {}, {}
            for s in SCORES:
                if s not in G_te:
                    continue
                stand[s] = evaluate(G_cal[s][cal_known], G_te[s], is_unk, pred_names, category[idx_te], benign,
                                    args.target)
                guard[s] = guard_eval(alarm_cal, alarm_te, pred_cal, pred_te, b, cal_ben, is_unk, is_ben,
                                      G_cal[s], G_te[s])
            res[mseed][fam] = {"standalone": stand, "guard": guard, "seconds": {k: round(v, 1) for k, v in times.items()},
                               "n_unknown": int(is_unk.sum())}
            print(f"[tohum {mseed}] {fam}: " + "  ".join(f"{s} {stand[s]['auroc']:.3f}" for s in stand)
                  + f"  | q=%5 kaçış: " + "  ".join(f"{s} {guard[s]['q0.05']['silent']:.4f}" for s in guard)
                  + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)

    # özet: aile ortalaması (tohum başına), sonra tohumlar arası ortalama ± sd
    def fam_mean(seed, fn):
        return float(np.mean([fn(res[seed][f]) for f in families]))

    summary = {}
    for s in SCORES:
        if not all(s in res[sd][families[0]]["standalone"] for sd in seeds):
            continue
        metr = {"auroc": lambda r: r["standalone"][s]["auroc"],
                "detect@cal": lambda r: r["standalone"][s]["detect@cal"],
                "known_reject@cal": lambda r: r["standalone"][s]["known_reject@cal"],
                "silent_benign@cal": lambda r: r["standalone"][s]["silent_benign@cal"]}
        for q in QS:
            metr[f"guard_silent_q{q}"] = (lambda qq: lambda r: r["guard"][s][f"q{qq}"]["silent"])(q)
            metr[f"guard_review_q{q}"] = (lambda qq: lambda r: r["guard"][s][f"q{qq}"]["review"])(q)
        per_seed = {k: [fam_mean(sd, fn) for sd in seeds] for k, fn in metr.items()}
        summary[s] = {k: {"mean": round(float(np.mean(v)), 5),
                          "sd": round(float(np.std(v, ddof=1)), 5) if len(v) > 1 else None,
                          "per_seed": [round(x, 5) for x in v]} for k, v in per_seed.items()}
    for s in summary:
        if s != REF_ALARM:
            summary[s]["diff_auroc_vs_tabm_disagree"] = [round(a - r, 4) for a, r in
                                                        zip(summary[s]["auroc"]["per_seed"], summary[REF_ALARM]["auroc"]["per_seed"])]
        if s != REF_GUARD:
            summary[s]["diff_guard_silent_q0.05_vs_tabm_energy"] = [
                round(a - r, 5) for a, r in zip(summary[s]["guard_silent_q0.05"]["per_seed"],
                                                summary[REF_GUARD]["guard_silent_q0.05"]["per_seed"])]
    out = {"dataset_id": args.dataset, "condition": args.condition, "dropped_features": drop, "families": families,
           "seeds": seeds,
           "status": "Analiz planı sonuçlardan önce yazıldı (README, Literatür karşılaştırması); eşik yok",
           "settings": {k: v for k, v in vars(args).items() if k != "data_file"},
           "summary": summary, "per_seed": {str(k): v for k, v in res.items()},
           "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    path = os.path.join(args.out_dir, f"open_set_baselines_{args.condition}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    plot(summary, families, os.path.join(args.out_dir, f"fig_open_set_baselines_{args.dataset}_{args.condition}.png"),
         spec["name"], args.condition, len(seeds))
    print("\nÖzet (aile ortalaması, tohumlar arası ort. ± sd):")
    for s, sm in summary.items():
        print(f"  {s:13s} AUROC {sm['auroc']['mean']:.3f} ± {sm['auroc']['sd'] or 0:.3f}   kontrol q=%5: kaçış "
              f"{sm['guard_silent_q0.05']['mean']:.4f} / inceleme {sm['guard_review_q0.05']['mean']:.4f}")
    print(f"\nÇıktı: {path}  ({out['runtime_seconds']:.0f} s)")


def plot(summary, families, path, name, cond, n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"tabm_disagree": "TabM anlaşmazlık", "tabm_energy": "TabM enerji", "tabm_msp": "TabM MSP",
             "knn": "kNN", "openmax": "OpenMax", "mahalanobis": "Mahalanobis", "autoencoder": "Otokodlayıcı"}
    ref = {"tabm_disagree", "tabm_energy", "tabm_msp", "knn"}
    keys = sorted(summary, key=lambda s: -summary[s]["auroc"]["mean"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    y = np.arange(len(keys))
    a1.barh(y, [summary[s]["auroc"]["mean"] for s in keys], xerr=[summary[s]["auroc"]["sd"] or 0 for s in keys],
            color=["#9aa5b1" if s in ref else "#2a78d6" for s in keys], capsize=2)
    a1.set_yticks(y)
    a1.set_yticklabels([names[s] for s in keys], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0.4, 1.0)
    a1.axvline(0.5, color="#555", ls=":", lw=0.8)
    a1.set_xlabel("AUROC (bilinmeyen / bilinen)", fontsize=8)
    a1.set_title("Tek başına bilinmeyen skoru", fontsize=9)
    for s in summary:
        rv = [summary[s][f"guard_review_q{q}"]["mean"] * 100 for q in QS]
        sl = [summary[s][f"guard_silent_q{q}"]["mean"] * 100 for q in QS]
        a2.plot(rv, sl, marker="o", lw=2.2 if s == REF_GUARD else 1.2, label=names[s],
                color=None if s not in ref else {"tabm_disagree": "#9aa5b1", "tabm_energy": "#14202b",
                                                 "tabm_msp": "#c3cad3", "knn": "#d9822b"}[s])
    a2.set_xlabel("Analiste giden normal trafik (%)", fontsize=8)
    a2.set_ylabel("Sessiz kaçış (%)", fontsize=8)
    a2.set_title("BENIGN kontrolü olarak (alarm: TabM anlaşmazlığı), q = %1 / 5 / 10", fontsize=9)
    a2.legend(frameon=False, fontsize=7)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
    fig.suptitle(f"{name}, koşul {cond}: literatür karşılaştırması ({len(families)} aile, {n_seeds} tohum)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
