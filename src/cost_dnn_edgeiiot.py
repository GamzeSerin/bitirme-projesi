"""Maliyet analizi: modeller ve DSS için başarı-maliyet ödünleşimi (ekip ortak aracı, --dataset; adı tarihsel).

Asıl deneyde (Edge-IIoTset: model_dnn_edgeiiot.py, CICIoT2023: model_ml.py) eğitilip work dizinine
kaydedilen modeller yüklenir; yeniden eğitim yapılmaz. Aynı veri hazırlığı ve aynı 80/20 bölme yeniden
kurulur. Veri setine özel her şey (yükleyici, resmi model dosyalarını bulma, protokol gereği atılan
sütunlar, donanım notu) src/common/open_set_registry.py'de. Yeniden kurulan özellikler manifest'teki
özelliklerle birebir aynı değilse ya da config.json kayıtla çelişiyorsa durur.

Ölçülenler (her model için):
  - Eğitim süresi: asıl deneyden (metrics.json); donanım manifest'te
  - Model boyutu: joblib dosyası (MB)
  - Toplu tahmin: 10.000 satır, CPU tek çekirdek (adil karşılaştırma); XGB ve TabM için ayrıca GPU
  - Tek paket gecikmesi: 300 tekil tahmin, CPU tek çekirdek; medyan ve %95'lik dilim (ms)
  - Saniyede paket (tek çekirdek, toplu)
DSS:
  - Oylayıcı alt kümeleri (DT, RF, XGB, LGBM, TabM'in tüm alt kümeleri) OOF tahminleriyle
    değerlendirilir; seçim YALNIZCA OOF macro-F1'e göre yapılır, test yalnızca raporlanır.
  - Her alt küme için: OOF ve test macro-F1, ALLOW edilen saldırı, yanlış alarm, gecikme toplamı.

Çıktı: outputs/<dataset>/cost/ (cost.json + fig_cost_f1_vs_latency_<dataset>.png)

Kullanım (asıl deneyin work dizini olan makinede; ~10-15 dk):
  python src/cost_dnn_edgeiiot.py --data-file "<.../DNN-EdgeIIoT-dataset.csv>" \\
      --work-dir /kaggle/working/work/dnn_edgeiiot
  python src/cost_dnn_edgeiiot.py --dataset dataset_23 --data-file data/dataset_23 \\
      --run-dir work/dataset_23/<çalıştırma>
"""

import argparse
import glob
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES, SEVERITY

OUT_DIR = "outputs/dnn_edgeiiot/cost"  # Edge-IIoTset varsayılanı (geriye uyumluluk)
DSS_CANDIDATES = ["DT", "RF", "XGB", "LGBM", "TabM"]


# ---------------------------------------------------------------- yardımcılar
def find_run_dir(work_dir):
    runs = [d for d in glob.glob(os.path.join(work_dir, "*")) if os.path.isdir(os.path.join(d, "models"))]
    if not runs:
        raise FileNotFoundError(f"{work_dir} altında models/ içeren bir çalıştırma yok "
                                "(asıl deney bu oturumda çalıştırılmalı)")
    return max(runs, key=os.path.getmtime)


def to_cpu_single_thread(name, model):
    """Adil karşılaştırma için: CPU, tek iş parçacığı."""
    inner = getattr(model, "estimator_", None)
    if name == "RF":
        model.set_params(n_jobs=1)
    elif name == "XGB" and inner is not None:
        inner.set_params(device="cpu", n_jobs=1)
    elif name == "LGBM" and inner is not None:
        inner.set_params(n_jobs=1)
    elif name == "TabM":
        import torch
        torch.set_num_threads(1)
        model.model_.to("cpu")
    return model


def to_gpu(name, model):
    """GPU'da çalışabilen modeller için; değilse None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None
    if name == "XGB":
        model.estimator_.set_params(device="cuda")
        return model
    if name == "TabM":
        model.model_.to("cuda")
        return model
    return None


def time_batch(model, X, repeats=3):
    model.predict_proba(X[:100])                     # ısınma
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_single(model, X, n=300):
    model.predict_proba(X[:1])
    t = []
    for i in range(n):
        row = X[i:i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def class_f1(y, proba, n_classes):
    return f1_score(y, proba.argmax(1), average=None, labels=range(n_classes), zero_division=0)


def dss_eval(voters, oof, test, ytr, yte, classes, decide):
    """main.py ile aynı kural: sınıf başına OOF F1 ağırlıklı birleşim, aksiyon ciddiyet toplamından."""
    n = len(classes)
    W = np.array([class_f1(ytr, oof[m], n) for m in voters])
    W = W / np.clip(W.sum(0, keepdims=True), 1e-9, None)

    def combine(probas):
        s = np.einsum("mc,mnc->nc", W, np.stack(probas))
        return s / np.clip(s.sum(1, keepdims=True), 1e-9, None)

    s_oof = combine([oof[m] for m in voters])
    s_te = combine([test[m] for m in voters])
    sev = np.array([SEVERITY[c] for c in classes])
    top = s_te.argmax(1)
    sev_conf = np.array([s_te[i, sev == sev[j]].sum() for i, j in enumerate(top)])
    actions = np.array([decide(classes[j], c) for j, c in zip(top, sev_conf)])
    benign = classes.index("BENIGN")
    attack = yte != benign
    alarm = top != benign
    return {
        "oof_macro_f1": round(float(f1_score(ytr, s_oof.argmax(1), average="macro")), 4),
        "test_macro_f1": round(float(f1_score(yte, top, average="macro")), 4),
        "attacks_allowed": int(((actions == "ALLOW") & attack).sum()),
        "missed_attacks": int((~alarm & attack).sum()),
        "false_alarm_rate": round(float((alarm & ~attack).sum() / max((~attack).sum(), 1)), 5),
        "analyst_review": int((actions == "ANALYST_REVIEW").sum()),
    }


def plot(rows, dss_best, out_dir, dataset_id="dnn_edgeiiot", dataset_name="Edge-IIoTset"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, r in rows.items():
        x, y = r["single_ms_median_cpu1"], r["test_macro_f1"]
        ax.scatter(x, y, s=60, color="#2a78d6" if name != "TabM" else "#d9822b", zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    if dss_best:
        ax.scatter(dss_best["single_ms_sum"], dss_best["test_macro_f1"], s=90, marker="*",
                   color="#0f6c7a", zorder=4)
        ax.annotate(f"DSS ({'+'.join(dss_best['voters'])})",
                    (dss_best["single_ms_sum"], dss_best["test_macro_f1"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=9, color="#0f6c7a")
    ax.set_xscale("log")
    ax.set_xlabel("Tek paket gecikmesi, CPU tek çekirdek (ms, medyan; log ölçek)")
    ax.set_ylabel("Test macro-F1")
    ys = [r["test_macro_f1"] for r in rows.values()] + ([dss_best["test_macro_f1"]] if dss_best else [])
    pad = max(0.003, 0.08 * (max(ys) - min(ys)))
    ax.set_ylim(min(ys) - pad, min(1.0, max(ys) + pad))
    ax.set_title(f"{dataset_name}: başarı – gecikme ödünleşimi (LR grafik dışında)", fontsize=11)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_cost_f1_vs_latency_{dataset_id}.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True, help="Edge-IIoTset: CSV; CICIoT2023: Merged*.csv klasörü")
    p.add_argument("--work-dir", default=None, help="Varsayılan Edge: /kaggle/working/work/dnn_edgeiiot, "
                                                     "diğerleri: work/<dataset>")
    p.add_argument("--run-dir", default=None, help="Belirtilmezse work-dir altındaki en yeni çalıştırma")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/cost")
    p.add_argument("--metrics", default=None, help="Varsayılan outputs/<dataset>/metrics.json")
    p.add_argument("--manifest", default=None, help="Varsayılan outputs/<dataset>/manifest.json")
    p.add_argument("--hardware-note", default=None, help="Varsayılan: kayıttaki not")
    p.add_argument("--batch-rows", type=int, default=10000)
    p.add_argument("--single-n", type=int, default=300)
    args = p.parse_args()
    spec = registry.get(args.dataset)
    base = f"outputs/{args.dataset}"
    args.work_dir = args.work_dir or ("/kaggle/working/work/dnn_edgeiiot" if args.dataset == "dnn_edgeiiot"
                                      else f"work/{args.dataset}")
    args.out_dir = args.out_dir or f"{base}/cost"
    args.metrics = args.metrics or f"{base}/metrics.json"
    args.manifest = args.manifest or f"{base}/manifest.json"
    os.makedirs(args.out_dir, exist_ok=True)
    import main as dss_main                             # decide() ve eşikler main.py'den

    run_dir = args.run_dir or find_run_dir(args.work_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    print(f"Çalıştırma: {run_dir}\n  ayarlar: {cfg}")
    # config.json kayıttaki veri hazırlığıyla aynı olmalı (yükleyici kayıttan geliyor)
    drop = sorted(f for f in str(cfg.get("drop_features") or "").split(",") if f)
    if drop != sorted(spec["protocol_drop"]) or cfg.get("keep_misaligned", False):
        raise SystemExit(f"Çalıştırmanın veri hazırlığı kayıtla çelişiyor: drop_features={drop} "
                         f"(kayıt {spec['protocol_drop']}), keep_misaligned={cfg.get('keep_misaligned', False)}")
    X, fine, category, _ = spec["loader"](args.data_file, cfg["cap"], cfg["seed"],
                                          drop_conflicts=bool(cfg["drop_conflicts"]))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    if list(X.columns) != list(manifest["features"]):
        raise SystemExit(f"Yeniden kurulan özellikler manifest'tekilerle aynı değil:\n  veri: {list(X.columns)}"
                         f"\n  manifest: {manifest['features']}")
    classes = [c for c in COMMON_CLASSES if c in set(category)]
    y = np.array([classes.index(c) for c in category])
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=fine,
                                      random_state=cfg["seed"])
    ytr, yte = y[idx_tr], y[idx_te]
    Xte = X.iloc[idx_te]
    Xb = Xte.iloc[:args.batch_rows]
    metrics = json.load(open(args.metrics)) if os.path.exists(args.metrics) else {}

    rows, oof, test = {}, {}, {}
    files = spec["model_files"](run_dir, manifest)
    print("Model dosyaları: " + ", ".join(f"{n}={os.path.basename(m)}" for n, (m, _) in files.items()))
    for name, (path, res_path) in files.items():
        res = joblib.load(res_path)
        oof[name], test[name] = res["oof_proba"], res["test_proba"]
        t0 = time.perf_counter()
        model = joblib.load(path)
        load_s = time.perf_counter() - t0
        model = to_cpu_single_thread(name, model)
        batch_s = time_batch(model, Xb)
        med, p95 = time_single(model, Xte, args.single_n)
        r = {
            "test_macro_f1": round(float(f1_score(yte, res["test_proba"].argmax(1), average="macro")), 4),
            "fit_seconds": metrics.get(name, {}).get("timing", {}).get("fit_seconds", res.get("fit_seconds")),
            "model_mb": round(os.path.getsize(path) / 1e6, 2),
            "load_seconds": round(load_s, 2),
            "batch_ms_per_1k_cpu1": round(1000 * batch_s / len(Xb) * 1000, 3),
            "packets_per_second_cpu1": int(len(Xb) / batch_s),
            "single_ms_median_cpu1": round(med, 3),
            "single_ms_p95_cpu1": round(p95, 3),
        }
        gpu_model = to_gpu(name, model)
        if gpu_model is not None:
            gb = time_batch(gpu_model, Xb)
            gmed, _ = time_single(gpu_model, Xte, min(args.single_n, 100))
            r.update(batch_ms_per_1k_gpu=round(1000 * gb / len(Xb) * 1000, 3),
                     single_ms_median_gpu=round(gmed, 3))
        rows[name] = r
        print(f"{name:5s} F1={r['test_macro_f1']:.4f}  {r['model_mb']:7.1f} MB  "
              f"toplu {r['batch_ms_per_1k_cpu1']:8.2f} ms/1k  tek {med:7.3f} ms (p95 {p95:.3f})"
              + (f"  GPU toplu {r['batch_ms_per_1k_gpu']:.2f} ms/1k" if "batch_ms_per_1k_gpu" in r else ""))
        del model

    # DSS: oylayıcı alt kümeleri, seçim OOF'a göre
    cands = [m for m in DSS_CANDIDATES if m in oof]
    subsets = []
    for k in range(1, len(cands) + 1):
        for voters in itertools.combinations(cands, k):
            e = dss_eval(list(voters), oof, test, ytr, yte, classes, dss_main.decide)
            e["voters"] = list(voters)
            e["single_ms_sum"] = round(sum(rows[v]["single_ms_median_cpu1"] for v in voters), 3)
            e["batch_ms_per_1k_sum"] = round(sum(rows[v]["batch_ms_per_1k_cpu1"] for v in voters), 3)
            subsets.append(e)
    subsets.sort(key=lambda e: (-e["oof_macro_f1"], e["batch_ms_per_1k_sum"]))
    best_oof = subsets[0]
    # OOF'ta en iyiye binde 1 yakın olanlar arasından en hızlısı (önceden belirlenmiş kural)
    near = [e for e in subsets if e["oof_macro_f1"] >= best_oof["oof_macro_f1"] - 0.001]
    fastest_near = min(near, key=lambda e: e["batch_ms_per_1k_sum"])
    current_rule = next(e for e in subsets if set(e["voters"]) == {
        m for m in cands if metrics.get(m, {}).get("oof", {}).get("macro_f1", 0)
        >= max(metrics.get(c, {}).get("oof", {}).get("macro_f1", 0) for c in cands) - dss_main.VOTER_MARGIN})

    print("\nDSS oylayıcıları (seçim OOF'a göre; test yalnızca raporlama):")
    for tag, e in [("şu anki kural", current_rule), ("OOF'ta en iyi", best_oof),
                   ("OOF'ta en iyiye ≤0,001 yakın, en hızlı", fastest_near)]:
        print(f"  {tag:40s} {'+'.join(e['voters']):24s} OOF {e['oof_macro_f1']:.4f}  test {e['test_macro_f1']:.4f}"
              f"  ALLOW saldırı {e['attacks_allowed']}  yanlış alarm {e['false_alarm_rate']:.4%}"
              f"  toplu {e['batch_ms_per_1k_sum']:.1f} ms/1k")

    out = {
        "dataset_id": args.dataset, "run_dir": os.path.basename(run_dir), "config": cfg,
        "model_files": {n: os.path.basename(m) for n, (m, _) in files.items()},
        "protocol": "Modeller asıl deneyden yüklendi. CPU ölçümleri tek iş parçacığıyla (adil karşılaştırma); "
                    "toplu tahmin 10.000 test satırı, 3 tekrarın medyanı; tek paket gecikmesi 300 tekil "
                    "tahminin medyanı ve %95'lik dilimi. DSS oylayıcı seçimi yalnızca OOF macro-F1 ile; "
                    "kural: OOF'ta en iyiye ≤0,001 yakın alt kümeler arasından en hızlısı.",
        "hardware_note": args.hardware_note or spec["hardware_note"],
        "gpu": manifest.get("runtime", {}).get("gpu"),
        "models": rows,
        "dss": {"current_rule": current_rule, "best_oof": best_oof,
                "fastest_near_best_oof": fastest_near, "all_subsets": subsets},
    }
    with open(os.path.join(args.out_dir, "cost.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    plot({k: v for k, v in rows.items() if k != "LR"}, fastest_near, args.out_dir, args.dataset, spec["name"])
    print(f"Çıktılar: {args.out_dir}")


if __name__ == "__main__":
    main()
