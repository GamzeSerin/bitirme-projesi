"""Açıklama ve derin öğrenme analizlerini sırayla çalıştıran ortak betik (ekip ortak aracı, --dataset).

Edge-IIoTset'te yapılan analizlerin aynısını başka bir veri setinde tek komutla koşmak için. Her adım önce
--quick ile kısa bir tesisat denemesi yapar; deneme başarısızsa durur, tam koşuya geçmez. Kayıtlar
outputs/<dataset>/logs/ altına yazılır. Analiz planları ve Edge-IIoTset sonuçları: data/dnn_edgeiiot/README.md.

Adımlar (--steps ile seçilir; sıra önemli):
  xai2    Bilinmeyen Kararının Açıklanması (XAI-2): src/xai_unknown_decision.py
          Önkoşul: dss_guard.py bu veri setinde koşmuş olmalı (outputs/<dataset>/dss_guard/rows_*.npz).
  dl      Derin öğrenme karşılaştırması: src/two_signal.py --classifier MLP / MCDropout, model tohumları
          --seeds; ardından src/dl_compare.py. TabM tarafı mevcut two_signal koşularından okunur
          (outputs/<dataset>/two_signal/, s1/, s2/; AUROC için satır dosyaları gerekir).
  xai3    Kaçışların Karşı-Olgusal Açıklaması (XAI-3) + eş kontrolü: src/xai_escape_counterfactual.py
          Önkoşul: xai2 (kayıtlı TabM ve Shapley değerleri).
  timing  Tahmin süresi: src/inference_time.py (--timing-family ailesi; önkoşul xai2).
  baselines Literatür karşılaştırması: src/open_set_baselines.py (OpenMax, Mahalanobis, otokodlayıcı; tek başına ve
          BENIGN kontrolü olarak; model tohumları --seeds). GPU önerilir.
  noise   Etiket karışıklığı payı: src/label_noise_check.py (ana koşulda başka etiketle birebir aynı vektöre düşen satırlar). CPU.
  time    Zaman bölmesinde iki sinyal kuralı: src/two_signal.py --split time --skip-cp, tohumlar --seeds. VARSAYILANDA YOK:
          yalnız yükleyicinin döndürdüğü satır sırası kayıt (zaman) sırasıysa anlamlıdır; önce bunu kontrol edin
          (Edge-IIoTset'te CSV satır sırası etiket içinde kayıt sırasıyla aynıydı).

Örnek (CICIoT2023):
  python src/xai_dl_suite.py --dataset dataset_23 --data-file data/dataset_23 \\
      --run-dir work/dataset_23/<asıl deney> --params-from run --families RECON,SPOOFING,WEB,BRUTEFORCE
  python src/xai_dl_suite.py ... --steps dl          # yalnız bir adım
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["xai2", "dl", "xai3", "timing", "baselines", "noise", "time"]


def run(cmd, log=None):
    """Komutu çalıştırır, çıktıyı ekrana (ve varsa kayıt dosyasına) yazar; başarılıysa True."""
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                sys.stdout.write(line)
                fh.write(line)
            code = p.wait()
    else:
        code = subprocess.run(cmd).returncode
    print(f"  -> {'tamam' if code == 0 else 'HATA (kod ' + str(code) + ')'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return code == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--run-dir", default=None, help="Asıl deneyin klasörü (--params-from run için)")
    p.add_argument("--params-from", choices=["run", "metrics"], default="run")
    p.add_argument("--condition", default=None, help="Varsayılan: kayıttaki ana koşul")
    p.add_argument("--families", required=True, help="XAI adımlarının aileleri, ör. RECON,SPOOFING,WEB,BRUTEFORCE")
    p.add_argument("--steps", default="xai2,dl,xai3,timing,baselines,noise", help=f"Virgülle: {','.join(STEPS)}")
    p.add_argument("--seeds", default="42,1,2", help="dl adımının model tohumları")
    p.add_argument("--cp-config", default="", help="two_signal için (MLP/MCDropout'ta kullanılmaz, iletilir)")
    p.add_argument("--timing-family", default=None, help="timing adımının ailesi (varsayılan: --families'in ilki)")
    p.add_argument("--skip-quick", action="store_true", help="Hızlı denemeleri atla (önerilmez)")
    args = p.parse_args()

    steps = [s for s in args.steps.split(",") if s]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"Bilinmeyen adım: {bad} (geçerli: {STEPS})")
    if args.params_from == "run" and not args.run_dir and any(s in steps for s in ("xai2", "xai3")):
        raise SystemExit("--params-from run için --run-dir gerekli (ya da --params-from metrics)")
    py = sys.executable
    logs = os.path.join("outputs", args.dataset, "logs")
    quick_dir = os.path.join("outputs_quick", args.dataset)
    base = ["--dataset", args.dataset, "--data-file", args.data_file]
    cond = ["--condition", args.condition] if args.condition else []
    params = ["--params-from", args.params_from] + (["--run-dir", args.run_dir] if args.run_dir else [])
    sys.path.insert(0, HERE)
    from common import open_set_registry as registry
    c = args.condition or registry.get(args.dataset)["default_condition"]
    t_start = time.perf_counter()

    def step(name, cmd, log, quick_extra):
        if not args.skip_quick:
            if not run(cmd + ["--quick"] + quick_extra):
                raise SystemExit(f"{name}: hızlı deneme başarısız; tam koşuya geçilmedi. Hata yukarıda.")
        if not run(cmd, os.path.join(logs, log)):
            raise SystemExit(f"{name}: tam koşu başarısız; kayıt: {os.path.join(logs, log)}")

    if "xai2" in steps:
        cmd = [py, os.path.join(HERE, "xai_unknown_decision.py")] + base + cond + params + ["--families", args.families]
        step("XAI-2", cmd, f"log_xai_unknown_decision_{c}.txt", ["--out-dir", os.path.join(quick_dir, "xai2")])

    if "dl" in steps:
        cp = ["--cp-config", args.cp_config] if args.cp_config else []
        for seed in [int(s) for s in args.seeds.split(",") if s]:
            for clf in ["MLP", "MCDropout"]:
                cmd = [py, os.path.join(HERE, "two_signal.py")] + base + cond + cp + ["--classifier", clf,
                                                                                      "--model-seed", str(seed)]
                sfx = clf.lower() + ("" if seed == 42 else f"_s{seed}")
                quick = ["--out-dir", os.path.join(quick_dir, f"two_signal_{sfx}")]
                # hızlı deneme yalnız ilk tohumda (tesisat her tohumda aynı)
                if seed != int(args.seeds.split(",")[0]):
                    if not run(cmd, os.path.join(logs, f"log_two_signal_{c}_{sfx}.txt")):
                        raise SystemExit(f"DL {clf} tohum {seed}: tam koşu başarısız")
                    continue
                step(f"DL {clf} tohum {seed}", cmd, f"log_two_signal_{c}_{sfx}.txt", quick)
        if not run([py, os.path.join(HERE, "dl_compare.py"), "--dataset", args.dataset, "--seeds", args.seeds] + cond):
            raise SystemExit("dl_compare başarısız")

    if "xai3" in steps:
        cmd = [py, os.path.join(HERE, "xai_escape_counterfactual.py")] + base + cond + params + ["--families", args.families]
        step("XAI-3", cmd, f"log_xai_escape_counterfactual_{c}.txt", ["--out-dir", os.path.join(quick_dir, "xai3")])

    if "timing" in steps:
        fam = args.timing_family or args.families.split(",")[0]
        cmd = [py, os.path.join(HERE, "inference_time.py")] + base + cond + ["--family", fam]
        step("Tahmin süresi", cmd, f"log_inference_time_{fam}_{c}.txt", ["--out-dir", os.path.join(quick_dir, "timing")])

    if "baselines" in steps:
        cmd = [py, os.path.join(HERE, "open_set_baselines.py")] + base + cond + ["--seeds", args.seeds]
        step("Literatür karşılaştırması", cmd, f"log_open_set_baselines_{c}.txt",
             ["--out-dir", os.path.join(quick_dir, "baselines")])

    if "noise" in steps:
        cmd = [py, os.path.join(HERE, "label_noise_check.py")] + base + cond
        step("Etiket karışıklığı", cmd, f"log_label_noise_{c}.txt", ["--out-dir", os.path.join(quick_dir, "noise")])

    if "time" in steps:
        for i, seed in enumerate(int(s) for s in args.seeds.split(",") if s):
            cmd = [py, os.path.join(HERE, "two_signal.py")] + base + cond + ["--split", "time", "--skip-cp",
                                                                            "--model-seed", str(seed)]
            log = f"log_two_signal_{c}_time" + ("" if seed == 42 else f"_s{seed}") + ".txt"
            if i == 0:
                step(f"Zaman bölmesi tohum {seed}", cmd, log, ["--out-dir", os.path.join(quick_dir, "two_signal_time")])
            elif not run(cmd, os.path.join(logs, log)):
                raise SystemExit(f"Zaman bölmesi tohum {seed}: tam koşu başarısız")

    print(f"\nHepsi bitti ({(time.perf_counter() - t_start) / 60:.0f} dk). Çıktılar: outputs/{args.dataset}/ "
          f"(xai_unknown_decision, two_signal, dl_compare, xai_escape_counterfactual, inference_time, open_set_baselines, label_noise); kayıtlar: {logs}")


if __name__ == "__main__":
    main()
