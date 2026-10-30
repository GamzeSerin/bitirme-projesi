"""Demo teşhisi, 3. adım: görülmemiş normal cihazın paketleri kullanım modelinin eğitim verisinde var mı?

Gözlem (30.10.2026, Mac, kullanım modeli, WEB senaryosu, ilk 20 bin paket): kurala hiç bakmadan seçilen üç normal cihazdan
Distance (%0,16 bağlantı analiste) ve Heart_Rate (%0) temiz; Modbus'ta paketlerin %79'u ALARM, içerikli paketlerin yalnız
%0,67'si NORMAL, bağlantı kuralıyla bağlantıların %12'si (içerikli bağlantıların %41'i) analiste gidiyor.
Hipotezler: (H1) Modbus normal trafiği eğitim örnekleminde çok az (NORMAL etiketinden 40.000 rastgele satır, cihazlar arasında
paylaşılıyor); (H2) mbtcp.trans_id her istekte artan bir sayaç (tcp.seq gibi): yeni kayıttaki değerler eğitimde görülmemiş
olabilir, vektörler bu yüzden tekil kalıyor.

Ölçülen:
  - PCAP: paket/içerikli paket sayısı, Modbus (mbtcp.len > 0) payı, trans_id aralığı.
  - Veri: ham NORMAL satırları ve kullanım örneklemindeki (load_usage_data, cap, tohum 42 = paketin eğitim havuzu) NORMAL
    satırları içinde Modbus satır sayısı ve trans_id aralığı.
  - Kapsama: PCAP'in içerikli paketlerinin koşul C vektörü (32 özellik) (a) ham NORMAL'de, (b) kullanım örnekleminde NORMAL'de
    birebir var mı; aynı ölçüm mbtcp.trans_id çıkarılınca (H2 sınaması). En sık içerikli paket tipleri (trans_id hariç) ve
    ham veride hangi etiketlerle geçtikleri.

Çalıştırma (Kaggle CPU, Edge-IIoTset ekli, tshark kurulu):
  python src/demo/check_normal_coverage_dnn_edgeiiot.py --pcap <Modbus pcap> --data-file <DNN-EdgeIIoT-dataset.csv>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from extractor_dnn_edgeiiot import CONDITIONS, FEATURES, extract, protocol_of

COLS_C = [f for f in FEATURES if f not in CONDITIONS["C_seq_ack_yok"]]
COLS_NO_TID = [c for c in COLS_C if c != "mbtcp.trans_id"]


def vec_hash(X, cols):
    return pd.util.hash_pandas_object(X[cols].astype(np.float32), index=False).to_numpy()


def coverage(Xp, Xd, cols):
    hd = set(vec_hash(Xd, cols)) if len(Xd) else set()
    return round(float(np.isin(vec_hash(Xp, cols), list(hd)).mean()), 4) if len(Xp) else None


def modbus_stats(X):
    m = X["mbtcp.len"] > 0
    t = X.loc[m, "mbtcp.trans_id"]
    return {"rows": int(len(X)), "modbus_rows": int(m.sum()), "modbus_share": round(float(m.mean()), 4) if len(X) else None,
            "trans_id_min": float(t.min()) if len(t) else None, "trans_id_max": float(t.max()) if len(t) else None,
            "trans_id_unique": int(t.nunique())}


def load_raw(data_file):
    from model_dnn_edgeiiot import LABEL, misaligned_mask
    df = pd.read_csv(data_file, low_memory=False, usecols=lambda c: c in FEATURES or c in (LABEL, "frame.time"))
    df = df[~misaligned_mask(df)]
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).to_numpy()
    return X[ok].reset_index(drop=True).astype(np.float32), df[LABEL].astype(str).str.strip().str.upper().to_numpy()[ok]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcap", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--max-packets", type=int, default=20000)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normal-label", default="NORMAL")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    name = os.path.splitext(os.path.basename(a.pcap))[0]
    out_path = a.out or f"outputs/dnn_edgeiiot/demo/normal_coverage_{name}.json"

    meta, X = extract(a.pcap, a.max_packets)
    keep = (protocol_of(meta) != "udp").to_numpy()
    Xp = X[keep].reset_index(drop=True)
    pay = (Xp["tcp.len"] > 0).to_numpy()
    out = {"pcap": os.path.basename(a.pcap), "packets_non_udp": int(len(Xp)), "content_packets": int(pay.sum()),
           "pcap_modbus": modbus_stats(Xp), "pcap_modbus_content": modbus_stats(Xp[pay])}
    print(f"{name}: {len(Xp)} paket (UDP hariç), içerikli {pay.sum()}; Modbus {out['pcap_modbus']}")

    Xr, lab = load_raw(a.data_file)
    nr = lab == a.normal_label
    from usage_data_dnn_edgeiiot import load_usage_data
    Xu, fine, _, _ = load_usage_data(a.data_file, a.cap, a.seed)
    Xu = Xu[FEATURES].astype(np.float32)
    nu = fine == a.normal_label
    out["data_normal_raw"] = modbus_stats(Xr[nr])
    out["data_normal_usage_sample"] = modbus_stats(Xu[nu])
    print(f"Ham NORMAL: {out['data_normal_raw']}")
    print(f"Kullanım örneklemi NORMAL (cap {a.cap}): {out['data_normal_usage_sample']}")

    Xpc = Xp[pay]
    cov = {}
    for tag, cols in (("C_32", COLS_C), ("C_without_trans_id", COLS_NO_TID)):
        cov[tag] = {"in_raw_normal": coverage(Xpc, Xr[nr], cols), "in_raw_any_label": coverage(Xpc, Xr, cols),
                    "in_usage_normal": coverage(Xpc, Xu[nu], cols)}
        print(f"İçerikli paketlerin kapsaması ({tag}): {cov[tag]}")
    out["content_coverage"] = cov

    hp = vec_hash(Xpc, COLS_NO_TID)
    hr = vec_hash(Xr, COLS_NO_TID)
    hu = vec_hash(Xu, COLS_NO_TID)
    vc = pd.Series(hp).value_counts().head(a.top)
    out["top_content_types_without_trans_id"] = []
    for h, n in vc.items():
        i = int(np.argmax(hp == h))
        row = {"packets": int(n), "share": round(n / len(hp), 4),
               "values": {c: float(Xpc.iloc[i][c]) for c in COLS_C if float(Xpc.iloc[i][c]) != 0},
               "raw_labels": {str(k): int(v) for k, v in pd.Series(lab[hr == h]).value_counts().head(5).items()},
               "usage_labels": {str(k): int(v) for k, v in pd.Series(fine[hu == h]).value_counts().head(5).items()}}
        out["top_content_types_without_trans_id"].append(row)
        print(f"\n{n} içerikli paket (%{n / len(hp) * 100:.1f}) {row['values']}")
        print(f"   ham veride: {row['raw_labels'] or 'YOK'}   kullanım örnekleminde: {row['usage_labels'] or 'YOK'}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nÇıktı: {out_path}")


if __name__ == "__main__":
    main()
