"""Kişi 3 - Edge-IIoTset sütun hizası / değer aralığı denetimi.

frame.time kontrolü yalnızca baştaki sütunlarda başlayan kaymayı yakalar. Bu betik her
ince etiket için her sütun grubunun beklenen biçimde olup olmadığını ölçer; böylece daha
sağdaki sütunlarda başlayan bir kayma ya da bozuk kayıt da görünür olur.

Kontroller (etiket başına ihlal oranı, %):
  zaman      frame.time bir zaman damgası mı
  ip         IP sütunlarında IPv4 adresi ya da 0 mı
  port       port sütunları 0-65535 arası tam sayı mı
  bayrak     0/1 bayrak sütunları gerçekten 0/1 mi
  arp        arp.opcode {0,1,2}, arp.hw.size {0,6} içinde mi
  sayisal    sayısal olması gereken sütunlarda metin var mı (sütun bazında, en kötü 5)

Kullanım (Kaggle CPU, ~2-3 dk; yalnızca okur, model eğitmez):
  python src/audit_dnn_edgeiiot.py --data-file "<.../DNN-EdgeIIoT-dataset.csv>"
Çıktı: outputs/dnn_edgeiiot/audit/audit.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from model_dnn_edgeiiot import LABEL, TIME_COL, TIME_PATTERN

IP_COLS = ["ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4"]
PORT_COLS = ["tcp.srcport", "tcp.dstport", "udp.port"]
FLAG_COLS = ["tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
             "tcp.connection.synack", "tcp.flags.ack", "dns.qry.qu", "dns.retransmission",
             "dns.retransmit_request", "mqtt.conflag.cleansess"]
# Metin olması doğal olan sütunlar (sayısal kontrolün dışında)
TEXT_OK = {TIME_COL, LABEL, "Attack_label", *IP_COLS, "http.file_data", "http.request.uri.query",
           "http.request.method", "http.referer", "http.request.full_uri", "http.request.version",
           "tcp.options", "tcp.payload", "dns.qry.name.len", "mqtt.conack.flags", "mqtt.msg",
           "mqtt.protoname", "mqtt.topic", "tcp.srcport"}
IPV4 = r"^\s*(\d{1,3}\.){3}\d{1,3}\s*$"


def num(s):
    return pd.to_numeric(s, errors="coerce")


def pct(mask):
    return round(100 * float(np.mean(mask)), 3) if len(mask) else 0.0


def audit_label(g):
    out = {"n": int(len(g))}
    out["zaman"] = pct(~g[TIME_COL].astype(str).str.match(TIME_PATTERN))
    ip = [c for c in IP_COLS if c in g]
    if ip:
        bad = np.zeros(len(g), bool)
        for c in ip:
            v = g[c].astype(str).str.strip()
            bad |= ~(v.str.match(IPV4) | v.isin(["0", "0.0"]))
        out["ip"] = pct(bad)
    bad = np.zeros(len(g), bool)
    for c in [c for c in PORT_COLS if c in g]:
        v = num(g[c])
        bad |= v.isna().to_numpy() | ((v < 0) | (v > 65535) | (v % 1 != 0)).to_numpy()
    out["port"] = pct(bad)
    bad = np.zeros(len(g), bool)
    for c in [c for c in FLAG_COLS if c in g]:
        bad |= ~num(g[c]).isin([0, 1]).to_numpy()
    out["bayrak"] = pct(bad)
    bad = np.zeros(len(g), bool)
    if "arp.opcode" in g:
        bad |= ~num(g["arp.opcode"]).isin([0, 1, 2]).to_numpy()
    if "arp.hw.size" in g:
        bad |= ~num(g["arp.hw.size"]).isin([0, 6]).to_numpy()
    out["arp"] = pct(bad)
    text = {}
    for c in g.columns:
        if c in TEXT_OK:
            continue
        r = pct(num(g[c]).isna() & g[c].notna())
        if r > 0:
            text[c] = r
    out["sayisal_en_kotu"] = dict(sorted(text.items(), key=lambda kv: -kv[1])[:5])
    out["sayisal"] = max(text.values()) if text else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True)
    p.add_argument("--out-dir", default="outputs/dnn_edgeiiot/audit")
    p.add_argument("--threshold", type=float, default=0.1, help="Raporlanacak en düşük ihlal oranı (%)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_file, low_memory=False, dtype=str)
    print(f"{len(df)} satır, {df.shape[1]} sütun")
    res = {lab: audit_label(g) for lab, g in df.groupby(LABEL)}
    del df

    checks = ["zaman", "ip", "port", "bayrak", "arp", "sayisal"]
    table = pd.DataFrame({lab: {k: r.get(k, 0.0) for k in ["n"] + checks} for lab, r in res.items()}).T
    table["n"] = table["n"].astype(int)
    print("\nİhlal oranı (%), etiket × kontrol:")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\n%{args.threshold}'in üzerindeki ihlaller:")
    flagged = False
    for lab, r in res.items():
        hits = {k: r[k] for k in checks if r.get(k, 0) > args.threshold}
        if hits:
            flagged = True
            print(f"  {lab:22s} {hits}")
            if r["sayisal"] > args.threshold:
                print(f"  {'':22s} metin içeren sayısal sütunlar: {r['sayisal_en_kotu']}")
    if not flagged:
        print("  Yok.")

    with open(os.path.join(args.out_dir, "audit.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset_id": "dnn_edgeiiot", "threshold_pct": args.threshold,
                   "checks": checks, "by_label": res}, f, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {args.out_dir}/audit.json")


if __name__ == "__main__":
    main()
