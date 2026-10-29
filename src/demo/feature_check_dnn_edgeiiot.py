"""Canlı demo, 1. adım: özellik ön kontrolü (Edge-IIoTset).

Soru: Veri setindeki bir PCAP'i kendi çıkarıcımızdan (tshark) geçirince, modelin kullandığı 34 özellik
veri setinin yayımlanan CSV'sindeki değerlerle aynı çıkıyor mu? Aynı çıkmazsa canlı demoda model,
eğitimde hiç görmediği bir temsile bakar.

Yöntem:
  1. Karşılaştırma CSV'si (arşivdeki ham saldırı/normal CSV'si, ör. "Attack traffic/XSS_attack.csv") okunur;
     sütun adları tshark alan adlarıdır.
  2. Aynı adlarla tshark -T fields çalıştırılır (bu tshark sürümünde olmayan alanlar raporlanır, 0 yazılır).
  3. Değerler normalleştirilir: onaltılık (0x..) -> tam sayı, True/False -> 1/0, boş -> 0, çok değerli
     alanlarda ilk değer. (Hangi kuralın gerektiği de raporlanır.)
  4. Karşılaştırma iki şekilde:
     a) Sıra hizası: satır sayıları eşitse aynı sıradaki paketler sütun sütun karşılaştırılır.
     b) Dağılım: her özelliğin değer kümesi ve sıklıkları (satır sayısı farklıysa da çalışır);
        ayrıca 34 özellikli vektörlerin kaçının CSV'de birebir karşılığı var.
  Çıktı: outputs/dnn_edgeiiot/demo/feature_check_<ad>.json ve ekrana özet.

Çalıştırma (Kaggle, Internet açık; tshark yoksa: apt-get install -y tshark):
  python src/demo/feature_check_dnn_edgeiiot.py --pcap "<.../XSS attacks.pcap>" --csv "<.../XSS_attack.csv>"
  ... --max-packets 50000   (büyük dosyalarda ilk N paket)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

FEATURES = ['arp.opcode', 'arp.hw.size', 'icmp.seq_le', 'icmp.transmit_timestamp', 'http.content_length',
            'http.response', 'tcp.ack', 'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
            'tcp.connection.synack', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq', 'udp.port',
            'udp.time_delta', 'dns.qry.name', 'dns.qry.qu', 'dns.retransmission', 'dns.retransmit_request',
            'dns.retransmit_request_in', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
            'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver', 'mbtcp.len', 'mbtcp.trans_id',
            'mbtcp.unit_id']   # outputs/dnn_edgeiiot/manifest.json, asıl deneyin 34 özelliği
LABEL_COLS = ["Attack_label", "Attack_type"]


def valid_fields(tshark):
    out = subprocess.run([tshark, "-G", "fields"], capture_output=True, text=True).stdout
    return {line.split("\t")[2] for line in out.splitlines() if line.startswith("F\t")}


def run_tshark(tshark, pcap, fields, max_packets):
    cmd = [tshark, "-r", pcap, "-T", "fields", "-E", "header=y", "-E", "separator=\t", "-E", "occurrence=f"]
    if max_packets:
        cmd += ["-c", str(max_packets)]
    for f in fields:
        cmd += ["-e", f]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("tshark hata verdi:\n" + r.stderr[-2000:])
    from io import StringIO
    return pd.read_csv(StringIO(r.stdout), sep="\t", dtype=str, keep_default_na=False, quoting=3)


def normalize(series, rules):
    """tshark metnini sayıya çevirir; kullanılan kuralları 'rules' kümesine ekler."""
    s = series.astype(str).str.strip().str.split(",").str[0].str.strip()   # çok değerli alan -> ilk değer
    s = s.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    out = pd.Series(np.zeros(len(s)), index=s.index)
    empty = s.eq("") | s.eq("nan")
    if (~empty).any():
        rules.add("boş -> 0") if empty.any() else None
    t = s[~empty]
    is_bool = t.isin(["True", "False"])
    if is_bool.any():
        rules.add("True/False -> 1/0")
        out[t.index[is_bool]] = t[is_bool].eq("True").astype(float)
    is_hex = t.str.match(r"^0x[0-9a-fA-F]+$")
    if is_hex.any():
        rules.add("onaltılık -> tam sayı")
        out[t.index[is_hex]] = t[is_hex].map(lambda v: float(int(v, 16)))
    rest = t.index[~is_bool & ~is_hex]
    num = pd.to_numeric(t[rest], errors="coerce")
    bad = num.isna()
    if bad.any():
        rules.add("sayı olmayan metin -> ayrıca raporlandı")
    out[rest] = num.fillna(np.nan).to_numpy()
    return out, t[rest][bad].head(5).tolist()


def csv_numeric(series, rules=None, bad_store=None, name=None):
    """CSV tarafına da aynı normalleştirme (onaltılık, True/False, çok değerli metinler)."""
    v, bad = normalize(series.fillna(""), rules if rules is not None else set())
    if bad and bad_store is not None:
        bad_store[name] = bad
    return v


def multiset_share(A, B):
    """A'daki vektörlerin B'de (sayıları da dikkate alarak) karşılığı olanların oranı; sıradan bağımsız."""
    from collections import Counter
    ca, cb = Counter(map(tuple, A)), Counter(map(tuple, B))
    return round(sum(min(n, cb[k]) for k, n in ca.items()) / max(len(A), 1), 4)


def tvd(a, b):
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    keys = pa.index.union(pb.index)
    return round(float(0.5 * (pa.reindex(keys, fill_value=0) - pb.reindex(keys, fill_value=0)).abs().sum()), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcap", required=True)
    p.add_argument("--csv", required=True, help="Karşılaştırma CSV'si (aynı kaydın veri setindeki hâli)")
    p.add_argument("--max-packets", type=int, default=0)
    p.add_argument("--tshark", default=shutil.which("tshark") or "tshark")
    p.add_argument("--out-dir", default="outputs/dnn_edgeiiot/demo")
    p.add_argument("--dnn-csv", default=None, help="Modelin eğitildiği DNN-EdgeIIoT-dataset.csv (asıl hedef temsil)")
    p.add_argument("--dnn-label", default=None, help="DNN dosyasında karşılaştırılacak Attack_type (ör. XSS, Normal)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.pcap))[0].replace(" ", "_")

    ver = subprocess.run([args.tshark, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"tshark: {ver}")
    ref = pd.read_csv(args.csv, low_memory=False, nrows=args.max_packets or None)
    cols = [c for c in ref.columns if c not in LABEL_COLS]
    valid = valid_fields(args.tshark)
    missing_fields = [c for c in cols if c not in valid]
    use = [c for c in cols if c in valid]
    print(f"CSV: {len(ref)} satır, {len(cols)} alan; bu tshark sürümünde olmayan alan: {missing_fields}")
    ours = run_tshark(args.tshark, args.pcap, use, args.max_packets)
    print(f"tshark: {len(ours)} paket")

    rules, report, nonnum, csv_rules, csv_nonnum = set(), {}, {}, set(), {}
    aligned = len(ours) == len(ref)
    for f in FEATURES:
        r = {}
        refv = csv_numeric(ref[f], csv_rules, csv_nonnum, f) if f in ref.columns else None
        if f not in use:
            r["status"] = "tshark'ta alan yok" if f in cols else "CSV'de sütun yok"
            report[f] = r
            continue
        ov, bad = normalize(ours[f], rules)
        if bad:
            nonnum[f] = bad
        r["ours_unique"] = int(ov.nunique())
        r["csv_unique"] = int(refv.nunique()) if refv is not None else None
        if aligned and refv is not None:
            a, b = ov.to_numpy(), refv.to_numpy()
            has = ~np.isnan(b)                       # bizde sayıya çevrilemeyen değer = uyuşmazlık
            eq = np.isclose(np.nan_to_num(a, nan=np.inf), b, rtol=1e-6, atol=1e-9) & ~np.isnan(a)
            r["aligned_exact_match"] = round(float(eq[has].mean()), 4) if has.any() else None
            r["ours_non_numeric_rows"] = int(np.isnan(a).sum())
            mism = np.where(has & ~eq)[0][:5]
            r["mismatch_examples"] = [{"row": int(i), "ours_raw": str(ours[f].iloc[i]), "ours": None if np.isnan(a[i]) else float(a[i]),
                                       "csv": float(b[i])} for i in mism]
        if refv is not None:
            r["tvd_vs_csv"] = tvd(ov.dropna().round(6), refv.dropna().round(6))
            top_o = ov.value_counts(normalize=True).head(5)
            top_c = refv.value_counts(normalize=True).head(5)
            r["top_values_ours"] = {str(k): round(float(v), 4) for k, v in top_o.items()}
            r["top_values_csv"] = {str(k): round(float(v), 4) for k, v in top_c.items()}
        report[f] = r

    # vektör düzeyi: 34 özellikli vektörlerin kaçının CSV'de birebir karşılığı var
    feats = [f for f in FEATURES if f in use and f in ref.columns]
    O = pd.DataFrame({f: normalize(ours[f], set())[0] for f in feats}).round(6)
    R = pd.DataFrame({f: csv_numeric(ref[f]) for f in feats}).round(6)
    vec_match = multiset_share(R.fillna(-1e18).to_numpy(), O.fillna(-2e18).to_numpy())   # CSV satırlarının bizde karşılığı

    # 4) asıl hedef: modelin eğitildiği DNN-EdgeIIoT dosyasındaki aynı etiketli satırlar
    dnn = None
    if args.dnn_csv:
        D = pd.read_csv(args.dnn_csv, usecols=lambda c: c in FEATURES or c == "Attack_type", low_memory=False)
        D = D[D["Attack_type"] == args.dnn_label]
        dnn_nonnum = {}
        Dn = pd.DataFrame({f: csv_numeric(D[f], set(), dnn_nonnum, f) for f in FEATURES if f in D.columns}).round(6)
        Of = pd.DataFrame({f: (O[f] if f in O.columns else pd.Series(0.0, index=O.index)) for f in Dn.columns})
        per = {f: {"tvd": tvd(Of[f].dropna(), Dn[f].dropna()),
                   "ours_top": {str(k): round(float(v), 4) for k, v in Of[f].value_counts(normalize=True).head(5).items()},
                   "dnn_top": {str(k): round(float(v), 4) for k, v in Dn[f].value_counts(normalize=True).head(5).items()}}
               for f in Dn.columns}
        dnn = {"file": os.path.basename(args.dnn_csv), "label": args.dnn_label, "rows": int(len(Dn)),
               "non_numeric_in_dnn": dnn_nonnum,
               "dnn_rows_found_in_ours": multiset_share(Dn.fillna(-1e18).to_numpy(), Of.fillna(-2e18).to_numpy()),
               "features": per}

    exact = {f: r.get("aligned_exact_match") for f, r in report.items() if r.get("aligned_exact_match") is not None}
    out = {"pcap": os.path.basename(args.pcap), "csv": os.path.basename(args.csv), "tshark": ver,
           "rows": {"ours": int(len(ours)), "csv": int(len(ref)), "aligned": aligned},
           "fields_missing_in_tshark": missing_fields, "normalization_rules_used": sorted(rules),
           "non_numeric_examples": nonnum, "csv_non_numeric_examples": csv_nonnum,
           "csv_normalization_rules_used": sorted(csv_rules),
           "csv_rows_found_in_ours_orderfree": vec_match, "features": report, "dnn_reference": dnn,
           "summary": {"features_compared": len(exact),
                       "features_exact_ge_0.99": sorted([f for f, v in exact.items() if v >= 0.99]),
                       "features_below_0.99": {f: v for f, v in sorted(exact.items(), key=lambda t: t[1]) if v < 0.99}}}
    path = os.path.join(args.out_dir, f"feature_check_{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nSatır: bizim {len(ours)} / CSV {len(ref)} (sıra hizası {'var' if aligned else 'YOK'})")
    print(f"Normalleştirme kuralları: {sorted(rules)}")
    if aligned:
        print(f"Birebir eşleşen özellik (>= %99): {len(out['summary']['features_exact_ge_0.99'])} / {len(exact)}")
        for f, v in out["summary"]["features_below_0.99"].items():
            ex = report[f]["mismatch_examples"][:2]
            print(f"  {f:28s} eşleşme {v:.3f}  örnek: {[(e['ours_raw'], e['csv']) for e in ex]}")
    print(f"CSV normalleştirme kuralları: {sorted(csv_rules)}")
    print(f"Ham CSV satırlarının bizde birebir karşılığı (sıradan bağımsız): {vec_match}")
    big = {f: r["tvd_vs_csv"] for f, r in report.items() if r.get("tvd_vs_csv", 0) > 0.01}
    print(f"Dağılımı ham CSV'den ayrışan özellikler (TVD > 0,01): {big}")
    if dnn:
        print(f"\nDNN dosyası, etiket {dnn['label']}: {dnn['rows']} satır; bizde birebir karşılığı olan: "
              f"{dnn['dnn_rows_found_in_ours']}; DNN'de sayı olmayan alanlar: {list(dnn['non_numeric_in_dnn'])}")
        for f, r in sorted(dnn["features"].items(), key=lambda t: -t[1]["tvd"]):
            if r["tvd"] > 0.01:
                print(f"  {f:26s} TVD {r['tvd']:.3f}  bizim {r['ours_top']}  DNN {r['dnn_top']}")
    for f, r in report.items():
        if "status" in r:
            print(f"  {f}: {r['status']}")
    print(f"Çıktı: {path}")


if __name__ == "__main__":
    sys.exit(main())
