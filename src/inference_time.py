"""Tahmin süresi: TabM (k = 32) / MLP (k = 1) / MC-dropout (T = 32) (ekip ortak aracı, --dataset).

Analiz planı sonuçlardan önce yazıldı (data/dnn_edgeiiot/README.md, "Tahmin süresi"); eşik yok.

Bir ailede (varsayılan WEB; LOFO bölmesi xai_unknown_decision ile aynı) aynı ön işlenmiş test satırları üzerinde
yalnız ağın ileri geçişi ölçülür (ön işleme üç modelde aynı olduğu için dışarıda):
  TabM         kayıtlı model (xai_unknown_decision/models), tek ileri geçiş, 32 üye birlikte
  MLP          aynı mimari k = 1, aynı tohumla burada eğitilir (two_signal --classifier MLP ile aynı)
  MC-dropout   aynı MLP; (a) sıralı: 32 ayrı geçiş (two_signal'deki uygulama),
               (b) vektörel: her satır 32 kez çoğaltılıp tek geçişte (dropout maskeleri satır başına bağımsız)
Ölçümler: toplu işlem (batch 8192, --n-rows satır) için satır/sn; tek satır gecikmesi (batch 1, --n-latency satır)
için satır başına ms; parametre sayısı. Her ölçüm --reps kez tekrarlanır, ısınma turundan sonra medyan alınır.
Cihazlar: GPU (varsa) ve CPU.

Çalıştırma:
  python src/inference_time.py --dataset dnn_edgeiiot --data-file <csv>
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
from sklearn.model_selection import train_test_split

from common import open_set_registry as registry
from common.taxonomy import COMMON_CLASSES


def timed(fn, reps, sync):
    fn()                                    # ısınma
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(t, 5) for t in ts]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dnn_edgeiiot", choices=list(registry.DATASETS))
    p.add_argument("--data-file", required=True)
    p.add_argument("--family", default="WEB")
    p.add_argument("--condition", default=None)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cal-size", type=float, default=0.1)
    p.add_argument("--passes", type=int, default=32)
    p.add_argument("--n-rows", type=int, default=10000)
    p.add_argument("--n-latency", type=int, default=200)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--model-dir", default=None, help="Varsayılan outputs/<dataset>/xai_unknown_decision/models")
    p.add_argument("--out-dir", default=None, help="Varsayılan outputs/<dataset>/inference_time")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.cap, args.n_rows, args.n_latency, args.reps = 3000, 500, 20, 2
    import torch
    from tabm_classifier import TabMClassifier

    spec = registry.get(args.dataset)
    cond = args.condition or spec["default_condition"]
    model_dir = args.model_dir or f"outputs/{args.dataset}/xai_unknown_decision/models"
    out_dir = args.out_dir or f"outputs/{args.dataset}/inference_time"
    os.makedirs(out_dir, exist_ok=True)

    X, fine, category, _ = spec["loader"](args.data_file, args.cap, args.seed)
    X, _ = registry.apply_condition(X, spec, cond)
    X = X.astype(np.float32)
    cats = set(category)
    present = [c for c in COMMON_CLASSES if c in cats] + sorted(cats - set(COMMON_CLASSES))
    fam_cats = spec["families"][args.family]
    known = [c for c in present if c not in fam_cats]
    idx_tr, idx_te = train_test_split(np.arange(len(category)), test_size=0.2, stratify=fine, random_state=args.seed)
    tr = idx_tr[~np.isin(category[idx_tr], fam_cats)]
    tr_fit, _ = train_test_split(tr, test_size=args.cal_size, stratify=fine[tr], random_state=args.seed)
    yfit = np.array([known.index(c) for c in category[tr_fit]])

    # modeller
    if args.quick:
        tabm = TabMClassifier(k=32, max_epochs=3, random_state=args.seed, verbose=False).fit(X.iloc[tr_fit], yfit)
        tabm_src = "quick: yeniden eğitildi"
    else:
        mp = os.path.join(model_dir, f"TabM_{args.family}_{cond}.joblib")
        if not os.path.exists(mp):
            raise SystemExit(f"{mp} yok: önce xai_unknown_decision.py koşmalı")
        tabm, tabm_src = joblib.load(mp), mp
    t0 = time.perf_counter()
    mlp = TabMClassifier(k=1, max_epochs=3 if args.quick else 60, random_state=args.seed,
                         verbose=False).fit(X.iloc[tr_fit], yfit)
    mlp_train = time.perf_counter() - t0
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(idx_te, min(args.n_rows, len(idx_te)), replace=False)
    Xt_tabm = tabm._prep(X.iloc[rows])
    Xt_mlp = mlp._prep(X.iloc[rows])
    n_params = {"TabM": int(sum(p.numel() for p in tabm.model_.parameters())),
                "MLP": int(sum(p.numel() for p in mlp.model_.parameters()))}
    print(f"{spec['name']} {args.family}, koşul {cond}: {len(rows)} satır; parametre {n_params}; "
          f"MLP eğitimi {mlp_train:.0f} s (bu cihazda)")

    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    results = {}
    for dev_name in devices:
        dev = torch.device(dev_name)
        sync = (lambda: torch.cuda.synchronize()) if dev_name == "cuda" else (lambda: None)
        nets = {"TabM": tabm.model_.to(dev).eval(), "MLP": mlp.model_.to(dev).eval()}
        drops = [m for m in nets["MLP"].modules() if isinstance(m, torch.nn.Dropout)]
        T = args.passes
        A = {"TabM": torch.as_tensor(Xt_tabm, device=dev), "MLP": torch.as_tensor(Xt_mlp, device=dev)}

        def mode(mc):
            nets["MLP"].eval()
            for m in drops:
                m.train(mc)

        def run(kind, x, batch):
            with torch.inference_mode():
                for i in range(0, len(x), batch):
                    xb = x[i:i + batch]
                    if kind == "TabM":
                        nets["TabM"](xb)
                    elif kind == "MLP":
                        nets["MLP"](xb)
                    elif kind == "MCDropout_sirali":
                        for _ in range(T):
                            nets["MLP"](xb)
                    else:                                          # MCDropout_vektorel
                        nets["MLP"](xb.repeat_interleave(T, 0)).view(len(xb), T, -1)

        res = {}
        torch.manual_seed(args.seed)
        for kind in ["TabM", "MLP", "MCDropout_sirali", "MCDropout_vektorel"]:
            src = "TabM" if kind == "TabM" else "MLP"
            mode(kind.startswith("MCDropout"))
            x = A[src]
            thr, thr_all = timed(lambda: run(kind, x, 8192), args.reps, sync)
            xl = x[:args.n_latency]
            lat, lat_all = timed(lambda: run(kind, xl, 1), args.reps, sync)
            res[kind] = {"throughput_rows_per_s": round(len(x) / thr, 1), "batch_seconds": round(thr, 5),
                         "latency_ms_per_row": round(1000 * lat / len(xl), 4),
                         "reps_batch": thr_all, "reps_latency": lat_all}
            print(f"  [{dev_name}] {kind:20s} toplu: {len(x) / thr:12,.0f} satır/sn   tek satır: "
                  f"{1000 * lat / len(xl):8.3f} ms")
        mode(False)
        results[dev_name] = res
        for n in nets.values():
            n.to("cpu")

    out = {"dataset_id": args.dataset, "family": args.family, "condition": cond, "tabm_model": tabm_src,
           "status": "Analiz planı sonuçlardan önce yazıldı (README, Tahmin süresi); eşik yok",
           "n_rows": int(len(rows)), "n_latency": int(min(args.n_latency, len(rows))), "passes": args.passes,
           "reps": args.reps, "n_params": n_params, "mlp_train_seconds": round(mlp_train, 1),
           "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads": torch.get_num_threads(), "results": results}
    path = os.path.join(out_dir, f"inference_time_{args.family}_{cond}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Çıktı: {path}")


if __name__ == "__main__":
    main()
