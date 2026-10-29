"""Canlı demo: PCAP -> modelin beklediği özellikler (Edge-IIoTset, tshark ile).

Özellik ön kontrolü (README, "Canlı demo, 1. adım") bu çıkarımın, modelin eğitildiği DNN-EdgeIIoT dosyasındaki
temsili birebir ürettiğini gösterdi (XSS: DNN satırlarının %100'ü). Bu modül o çıkarımın kalıcı hâlidir;
demonun arka tarafı ve doğrulama betiği (validate_extractor_dnn_edgeiiot.py) aynı kodu kullanır.

Kullanım (kütüphane):
    from demo.extractor_dnn_edgeiiot import extract, model_matrix
    meta, X = extract("XSS attacks.pcap", max_packets=50000)      # X: 34 özellik, float32
    Xc = model_matrix(X, "C_seq_ack_yok")                          # modelin girdisi: 32 sütun, eğitim sırası

Komut satırı (deneme):
    python src/demo/extractor_dnn_edgeiiot.py --pcap "<pcap>" --max-packets 1000 --out x.csv

Normalleştirme (veri setinin CSV'leriyle aynı): onaltılık -> tam sayı, True/False -> 1/0, boş -> 0, çok değerli alanda
ilk değer. Metin değerli alan (dns.qry.name) için kural --dns-policy / DNS_POLICY ile verilir; varsayılan "zero"
(metin -> 0) yalnızca geçicidir, DNN dosyasındaki değer kümesine göre kesinleşecek (README).
"""

import argparse
import os
import shutil
import subprocess
import sys
from io import StringIO

import numpy as np
import pandas as pd

# outputs/dnn_edgeiiot/manifest.json "features": eğitimdeki sütun SIRASI (model ham dizi alır, sıra önemli)
FEATURES = ['arp.opcode', 'arp.hw.size', 'icmp.seq_le', 'icmp.transmit_timestamp', 'http.content_length',
            'http.response', 'tcp.ack', 'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
            'tcp.connection.synack', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq', 'udp.port',
            'udp.time_delta', 'dns.qry.name', 'dns.qry.qu', 'dns.retransmission', 'dns.retransmit_request',
            'dns.retransmit_request_in', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
            'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver', 'mbtcp.len', 'mbtcp.trans_id',
            'mbtcp.unit_id']
# Gösterim ve süzme için (modele girmez)
META = ['frame.number', 'frame.time_epoch', 'frame.len', 'ip.src', 'ip.dst', 'tcp.srcport', 'udp.srcport',
        'udp.dstport', 'eth.type', 'ip.proto']
TEXT_FIELDS = ['dns.qry.name']
CONDITIONS = {"A_tam": [], "C_seq_ack_yok": ["tcp.seq", "tcp.ack"]}   # open_set_registry ile aynı (demoda C)


def check_manifest(path="outputs/dnn_edgeiiot/manifest.json"):
    """Sabit liste ile manifest aynı mı (sıra dahil); manifest yoksa sessizce geçer."""
    if os.path.exists(path):
        import json
        feats = json.load(open(path, encoding="utf-8"))["features"]
        if feats != FEATURES:
            raise SystemExit(f"Çıkarıcının özellik listesi manifest ile uyuşmuyor:\n{feats}")


def to_number(series, rules=None):
    """tshark metni -> sayı. Çevrilemeyen metin NaN kalır (çağıran karar verir)."""
    s = series.astype(str).str.strip().str.split(",").str[0].str.strip()
    s = s.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    empty = s.eq("") | s.eq("nan")
    out[empty] = 0.0
    t = s[~empty]
    is_bool = t.isin(["True", "False"])
    if is_bool.any():
        out.loc[t.index[is_bool]] = t[is_bool].eq("True").astype(float).to_numpy()
    is_hex = t.str.match(r"^0x[0-9a-fA-F]+$")
    if is_hex.any():
        out.loc[t.index[is_hex]] = np.array([float(int(v, 16)) for v in t[is_hex]])
    rest = t.index[~is_bool & ~is_hex]
    if len(rest):
        out.loc[rest] = pd.to_numeric(t[rest], errors="coerce").to_numpy(dtype=float)
    if rules is not None:
        if empty.any():
            rules.add("boş -> 0")
        if is_bool.any():
            rules.add("True/False -> 1/0")
        if is_hex.any():
            rules.add("onaltılık -> tam sayı")
    return out


def encode_text(series, policy):
    """Metin alanı (dns.qry.name) için kural. 'zero': her metin 0 (geçici); 'present': boş değilse 1;
    'length': karakter sayısı. Sayısal gelen değer olduğu gibi kalır."""
    s = series.astype(str).str.strip().str.split(",").str[0].str.strip()
    num = pd.to_numeric(s, errors="coerce")
    empty = s.eq("") | s.eq("nan")
    if policy == "zero":
        txt = pd.Series(0.0, index=s.index)
    elif policy == "present":
        txt = pd.Series(1.0, index=s.index)
    elif policy == "length":
        txt = s.str.len().astype(float)
    else:
        raise ValueError(f"Bilinmeyen metin kuralı: {policy}")
    return num.where(num.notna(), txt).where(~empty, 0.0)


def tshark_fields(tshark, pcap, fields, max_packets=0, display_filter=None):
    cmd = [tshark, "-r", pcap, "-n", "-T", "fields", "-E", "header=y", "-E", "separator=\t", "-E", "occurrence=a",
           "-E", "aggregator=,"]
    if max_packets:
        cmd += ["-c", str(max_packets)]
    if display_filter:
        cmd += ["-Y", display_filter]
    for f in fields:
        cmd += ["-e", f]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("tshark hata verdi:\n" + r.stderr[-2000:])
    df = pd.read_csv(StringIO(r.stdout), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    if len(df.columns) != len(fields):
        raise RuntimeError(f"tshark {len(df.columns)} sütun verdi, {len(fields)} bekleniyordu")
    df.columns = list(fields)   # başlıktaki yazım tshark sürümüne göre değişebilir; sıra -e sırasıyla aynı
    return df


def extract(pcap, max_packets=0, tshark=None, dns_policy=None, display_filter=None, rules=None):
    """PCAP -> (meta, X). X: FEATURES sırasıyla 34 sütun, float32, NaN yok."""
    tshark = tshark or shutil.which("tshark") or "tshark"
    dns_policy = dns_policy or os.environ.get("DNS_POLICY", "zero")
    raw = tshark_fields(tshark, pcap, META + FEATURES, max_packets, display_filter)
    X = pd.DataFrame(index=raw.index)
    for f in FEATURES:
        X[f] = encode_text(raw[f], dns_policy) if f in TEXT_FIELDS else to_number(raw[f], rules)
    bad = X.isna()
    if bad.any().any():   # tshark beklenmedik metin verdiyse: 0 yaz, ama say (sessiz geçmesin)
        X.attrs["non_numeric_counts"] = {c: int(n) for c, n in bad.sum().items() if n}
        X = X.fillna(0.0)
    meta = raw[META].copy()
    meta["frame.time_epoch"] = pd.to_numeric(meta["frame.time_epoch"], errors="coerce")
    meta["dns.qry.name_raw"] = raw["dns.qry.name"]
    return meta, X.astype(np.float32)


def model_matrix(X, condition="C_seq_ack_yok"):
    """Modelin girdisi: koşulun sütunları çıkarılmış, eğitim sırasıyla (manifest sırası korunur)."""
    return X[[f for f in FEATURES if f not in CONDITIONS[condition]]]


def protocol_of(meta):
    """Paket türü (süzme ve raporlama için): tcp / udp / icmp / arp / other."""
    p = pd.Series("other", index=meta.index)
    p[meta["eth.type"].str.contains("0x0806", na=False)] = "arp"
    proto = meta["ip.proto"].str.split(",").str[0]
    p[proto.eq("1")] = "icmp"
    p[proto.eq("17")] = "udp"
    p[proto.eq("6")] = "tcp"
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcap", required=True)
    p.add_argument("--max-packets", type=int, default=1000)
    p.add_argument("--condition", default="C_seq_ack_yok", choices=list(CONDITIONS))
    p.add_argument("--dns-policy", default=None, choices=["zero", "present", "length"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    check_manifest()
    rules = set()
    meta, X = extract(args.pcap, args.max_packets, dns_policy=args.dns_policy, rules=rules)
    Xc = model_matrix(X, args.condition)
    print(f"{len(X)} paket; model girdisi {Xc.shape[1]} sütun; kurallar {sorted(rules)}; "
          f"paket türleri {protocol_of(meta).value_counts().to_dict()}")
    if X.attrs.get("non_numeric_counts"):
        print(f"Sayıya çevrilemeyen (0 yazıldı): {X.attrs['non_numeric_counts']}")
    if args.out:
        pd.concat([meta, Xc], axis=1).to_csv(args.out, index=False)
        print(f"Çıktı: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
