"""Canlı demo, 2. adım: çıkarıcının doğrulanması ve kalan iki açık nokta (Edge-IIoTset).

Plan README'de sonuçlardan önce yazıldı ("Canlı demo, 2. adım"). Çıkarıcı (extractor_dnn_edgeiiot.py) birkaç PCAP'te
koşulur ve modelin eğitildiği DNN-EdgeIIoT dosyasındaki aynı etiketli satırlarla karşılaştırılır:
  - ours_in_dnn       : bizim satırlarımızın (34 özellik) DNN'de aynı etiketle birebir karşılığı olanların oranı
                        (DNN bir örneklem olduğu için asıl ölçü bu yön; tekrar eden vektörler küme olarak sayılır)
  - dnn_in_ours       : DNN satırlarının bizde karşılığı (yalnız PCAP tamamen okunduysa anlamlı)
  - eşleşmeyen satırlarımızın paket türü (tcp/udp/icmp/arp) ve en sık IP çiftleri  -> yazarların paket seçimi kuralı
  - özellik başına dağılım farkı (TVD)
Açık nokta 1 (paket seçimi): eşleşmeyenler belirli bir paket türünde/IP çiftinde toplanıyor mu?
Açık nokta 2 (dns.qry.name): DNN'deki değer kümesi etiket başına; bizim metin değerlerimizle yan yana.

Çalıştırma (Kaggle, CPU, Internet açık):
  python src/demo/validate_extractor_dnn_edgeiiot.py --root "<.../Edge-IIoTset dataset>" --max-packets 300000
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from extractor_dnn_edgeiiot import FEATURES, extract, protocol_of, to_number

# (PCAP, DNN Attack_type). Demoda kullanılacak aileler ağırlıkta: WEB (bilinmeyen), BRUTEFORCE, RECON, DDOS, normal.
PCAPS = [("Attack traffic/XSS attacks.pcap", "XSS"),
         ("Attack traffic/SQL injection attack.pcap", "SQL_injection"),
         ("Attack traffic/Uploading attack.pcap", "Uploading"),
         ("Attack traffic/Password attacks.pcap", "Password"),
         ("Attack traffic/Port Scanning attack.pcap", "Port_Scanning"),
         ("Attack traffic/DDoS TCP SYN Flood Attacks.pcap", "DDoS_TCP"),
         ("Normal traffic/Distance/Distance.pcap", "Normal"),
         ("Normal traffic/Temperature_and_Humidity/Temperature_and_Humidity.pcap", "Normal")]


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    k = pa.index.union(pb.index)
    return round(float(0.5 * (pa.reindex(k, fill_value=0) - pb.reindex(k, fill_value=0)).abs().sum()), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="'Edge-IIoTset dataset' klasörü")
    p.add_argument("--dnn-csv", default=None, help="Varsayılan <root>/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv")
    p.add_argument("--max-packets", type=int, default=300000)
    p.add_argument("--only", default=None, help="Virgülle etiketler (ör. XSS,Normal)")
    p.add_argument("--out-dir", default="outputs/dnn_edgeiiot/demo")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dnn_path = args.dnn_csv or os.path.join(args.root, "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv")

    print(f"Okunuyor: {dnn_path}")
    D = pd.read_csv(dnn_path, usecols=lambda c: c in FEATURES or c == "Attack_type",
                    dtype={"dns.qry.name": str, "Attack_type": str}, low_memory=False)
    labels = sorted(D["Attack_type"].unique())
    print(f"DNN: {len(D)} satır; etiketler {labels}")

    # Açık nokta 2: dns.qry.name'in DNN'deki ham değerleri (etiket başına)
    dns_profile = {}
    for lab, g in D.groupby("Attack_type"):
        vc = g["dns.qry.name"].value_counts(normalize=True)
        dns_profile[lab] = {"n_unique": int(g["dns.qry.name"].nunique()),
                            "top": {str(k): round(float(v), 4) for k, v in vc.head(8).items()}}
    print("dns.qry.name (DNN, etiket başına en sık değerler):")
    for lab, d in dns_profile.items():
        print(f"  {lab:22s} {d['n_unique']:4d} farklı  {d['top']}")

    DN = {}
    for lab, g in D.groupby("Attack_type"):
        DN[lab] = pd.DataFrame({f: (to_number(g[f]) if f == "dns.qry.name" else pd.to_numeric(g[f], errors="coerce"))
                                for f in FEATURES}).round(6).fillna(-1e18)
    del D

    only = set(args.only.split(",")) if args.only else None
    results = []
    for rel, lab in PCAPS:
        if only and lab not in only:
            continue
        path = os.path.join(args.root, rel)
        if not os.path.exists(path):
            hits = glob.glob(os.path.join(args.root, "**", os.path.basename(path)), recursive=True)
            if not hits:
                print(f"\n[{lab}] PCAP yok: {rel}")
                continue
            path = hits[0]
        if lab not in DN:
            print(f"\n[{lab}] DNN'de bu etiket yok")
            continue
        meta, X = extract(path, args.max_packets)
        Xr = X.astype(float).round(6)
        full = len(X) < args.max_packets
        dset = set(map(tuple, DN[lab].to_numpy()))
        ours_keys = [tuple(v) for v in Xr.to_numpy()]
        hit = np.array([k in dset for k in ours_keys])
        cd, co = Counter(map(tuple, DN[lab].to_numpy())), Counter(ours_keys)
        dnn_in_ours = sum(min(n, co[k]) for k, n in cd.items()) / max(len(DN[lab]), 1)
        proto = protocol_of(meta)
        miss = ~hit
        pair = (meta["ip.src"].str.split(",").str[0] + " > " + meta["ip.dst"].str.split(",").str[0]).where(
            meta["ip.src"].ne(""), "(IP yok)")
        r = {"pcap": os.path.basename(path), "label": lab, "packets": int(len(X)), "pcap_fully_read": full,
             "dnn_rows": int(len(DN[lab])),
             "ours_in_dnn": round(float(hit.mean()), 4),
             "dnn_in_ours": round(float(dnn_in_ours), 4) if full else None,
             "by_protocol": {pr: {"packets": int((proto == pr).sum()),
                                  "in_dnn": round(float(hit[(proto == pr).to_numpy()].mean()), 4)}
                             for pr in sorted(proto.unique())},
             "unmatched_top_pairs": {k: int(v) for k, v in pair[miss].value_counts().head(8).items()},
             "matched_top_pairs": {k: int(v) for k, v in pair[hit].value_counts().head(5).items()},
             "tvd": {f: tvd(Xr[f], DN[lab][f].replace(-1e18, np.nan).dropna()) for f in FEATURES},
             "dns_text_values_ours": {k: int(v) for k, v in
                                      meta.loc[meta["dns.qry.name_raw"].ne(""), "dns.qry.name_raw"]
                                      .value_counts().head(8).items()},
             "non_numeric": X.attrs.get("non_numeric_counts", {})}
        results.append(r)
        print(f"\n[{lab}] {r['pcap']}: {r['packets']} paket ({'tamamı' if full else 'ilk ' + str(args.max_packets)}); "
              f"DNN {r['dnn_rows']} satır")
        print(f"  bizim satırların DNN'de karşılığı: {r['ours_in_dnn']:.4f}   DNN satırlarının bizde: {r['dnn_in_ours']}")
        print(f"  paket türüne göre (sayı, DNN'de karşılığı): "
              + "  ".join(f"{k} {v['packets']} ({v['in_dnn']:.3f})" for k, v in r["by_protocol"].items()))
        print(f"  eşleşmeyen en sık IP çiftleri: {r['unmatched_top_pairs']}")
        print(f"  eşleşen en sık IP çiftleri:    {r['matched_top_pairs']}")
        big = {f: v for f, v in r["tvd"].items() if v > 0.01}
        print(f"  TVD > 0,01: {big}")
        if r["dns_text_values_ours"]:
            print(f"  bizdeki dns.qry.name metinleri: {r['dns_text_values_ours']}")

    out = {"dnn_file": os.path.basename(dnn_path), "max_packets": args.max_packets, "labels_in_dnn": labels,
           "dns_qry_name_profile_dnn": dns_profile, "pcaps": results,
           "status": "Plan sonuçlardan önce yazıldı (README, Canlı demo, 2. adım)"}
    path = os.path.join(args.out_dir, "validate_extractor.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {path}")


if __name__ == "__main__":
    sys.exit(main())
