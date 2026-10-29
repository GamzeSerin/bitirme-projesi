"""Canlı demo teşhisi: yerelde normal trafiğin beklenmedik alarm oranı tshark sürümünden mi, modelden mi?

Gözlem (29.10.2026, Mac, tshark 4.6.9): WEB senaryosunda Temperature_and_Humidity'nin ilk 20 bin paketinin %51'i ALARM, %12'si
SALDIRI aldı; deneylerde aynı modelin normal test trafiğinde alarm yükü ≈ %1. scikit-learn sürümü elendi (1.6.1 ile aynı sonuç).

İki adım:
  1) Kaggle'da (tshark 4.2.2, çıkarıcı orada doğrulandı) aynı PCAP'i çıkar ve kaydet:
       python src/demo/diagnose_replay_dnn_edgeiiot.py export --pcap <pcap> --out outputs/dnn_edgeiiot/demo/kaggle_extract/<ad>.csv.gz
  2) Yerelde karşılaştır (aynı PCAP, kendi tshark'ınla) ve modeli iki çıkarımda da çalıştır:
       python src/demo/diagnose_replay_dnn_edgeiiot.py compare --pcap <pcap> --ref <kaggle csv.gz> --bundle <paket> --scenario WEB
Rapor: tshark sürümleri; çerçeve numarasına göre hizalı, özellik başına uyuşmazlık oranı ve örnekleri; iki çıkarımda karar dağılımı;
alarm alan paketlerin en sık değerleri ve en sık "paket tipleri" (32 özellikli vektör) ile bunların karar dağılımı.
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from extractor_dnn_edgeiiot import CONDITIONS, FEATURES, extract, model_matrix, protocol_of


def tshark_version():
    t = shutil.which("tshark") or "tshark"
    return subprocess.run([t, "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def extract_table(pcap, max_packets):
    meta, X = extract(pcap, max_packets)
    keep = (protocol_of(meta) != "udp").to_numpy()
    out = pd.concat([meta[["frame.number"]].astype(int).reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return out[keep].reset_index(drop=True)


def cmd_export(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t = extract_table(a.pcap, a.max_packets)
    t.to_csv(a.out, index=False)
    print(f"{tshark_version()}\n{len(t)} paket (UDP atıldı) → {a.out}")


def profile(df, cols, label):
    print(f"\n  {label}: {len(df)} paket")
    for c in cols:
        vc = df[c].value_counts(normalize=True).head(4)
        print(f"    {c:24s} " + "  ".join(f"{k:g}: %{v * 100:.1f}" for k, v in vc.items()))


def cmd_compare(a):
    from replay_dnn_edgeiiot import Scenario
    print(f"Yerel: {tshark_version()}")
    loc = extract_table(a.pcap, a.max_packets)
    ref = pd.read_csv(a.ref)
    m = loc.merge(ref, on="frame.number", suffixes=("_yerel", "_kaggle"), how="outer", indicator=True)
    print(f"Paket: yerel {len(loc)}, Kaggle {len(ref)}; ikisinde de olan {int((m['_merge'] == 'both').sum())}")
    both = m[m["_merge"] == "both"]
    diff = {}
    for f in FEATURES:
        a_, b_ = both[f + "_yerel"].to_numpy(float), both[f + "_kaggle"].to_numpy(float)
        bad = ~np.isclose(a_, b_, rtol=1e-6, atol=1e-9)
        if bad.any():
            ex = both.loc[bad, ["frame.number", f + "_yerel", f + "_kaggle"]].head(3).to_numpy().tolist()
            diff[f] = (round(float(bad.mean()) * 100, 2), ex)
    print("Özellik başına uyuşmazlık (yerel tshark ↔ Kaggle tshark):")
    if not diff:
        print("  YOK: iki çıkarım birebir aynı")
    for f, (pct, ex) in sorted(diff.items(), key=lambda t: -t[1][0]):
        print(f"  {f:26s} %{pct:6.2f}  örnek [çerçeve, yerel, kaggle]: {ex}")

    scen = Scenario(a.bundle, a.scenario)
    cond = scen.meta["condition"]
    res = {}
    for name, tab in (("yerel", loc), ("Kaggle", ref)):
        dec = scen.decide(model_matrix(tab[FEATURES].astype(np.float32), cond))
        res[name] = (tab, dec)
        share = dec["decision"].value_counts(normalize=True)
        print(f"\nKarar dağılımı, {name} çıkarımı: " + "  ".join(f"{k} %{v * 100:.2f}" for k, v in share.items()))

    tab, dec = res["Kaggle" if a.use_ref else "yerel"]
    colsC = [f for f in FEATURES if f not in CONDITIONS[cond]]
    varying = [c for c in colsC if tab[c].nunique() > 1]
    print(f"\nModel girdisinde bu PCAP'te değişen özellikler: {varying}")
    for d in ("ALARM", "SALDIRI", "NORMAL"):
        sel = (dec["decision"] == d).to_numpy()
        if sel.any():
            profile(tab.loc[sel], varying, d)
    keys = tab[colsC].round(6).astype(str).agg("|".join, axis=1)
    g = pd.DataFrame({"key": keys, "decision": dec["decision"].to_numpy()})
    top = g["key"].value_counts().head(a.top)
    print(f"\nEn sık {a.top} paket tipi (32 özellikli vektör) ve kararları:")
    for k, n in top.items():
        ds = g.loc[g["key"] == k, "decision"].value_counts().to_dict()
        vals = dict(zip(colsC, k.split("|")))
        short = {c: vals[c] for c in varying}
        print(f"  {n:6d} paket (%{n / len(g) * 100:4.1f})  kararlar {ds}  {short}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--pcap", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-packets", type=int, default=20000)
    c = sub.add_parser("compare")
    c.add_argument("--pcap", required=True)
    c.add_argument("--ref", required=True, help="Kaggle'da export ile üretilen csv.gz")
    c.add_argument("--bundle", required=True)
    c.add_argument("--scenario", default="WEB")
    c.add_argument("--max-packets", type=int, default=20000)
    c.add_argument("--top", type=int, default=8)
    c.add_argument("--use-ref", action="store_true", help="Profil için Kaggle çıkarımını kullan (varsayılan yerel)")
    a = p.parse_args()
    cmd_export(a) if a.cmd == "export" else cmd_compare(a)


if __name__ == "__main__":
    main()
