"""Canlı demo, 3. adım: PCAP'leri oynat → karar ver (komut satırı; Edge-IIoTset).

Bir senaryo = bir "görülmemiş" aile: o aileyi HİÇ görmemiş TabM (demo paketinden) ve deneylerdeki iki sinyal kuralı.
Birden fazla PCAP aynı anda oynatılabilir (her biri kendi başlangıcından itibaren göreli zamanla, birleştirilerek):
normal trafik + bilinen saldırı + görülmemiş saldırı karışık akar.

Her paket için karar (iki sinyal kuralı, README "İki sinyalli karar kuralı"):
  ALARM     üye anlaşmazlığı > tau_alarm                      → analiste ("tanımadığım bir şey")
  KONTROL   alarm yok, tahmin BENIGN, enerji > tau_guard      → analiste ("normal dedim ama emin değilim")
  SALDIRI   alarm yok, tahmin bilinen bir saldırı sınıfı      → engelle / uyar (sınıfıyla)
  NORMAL    alarm yok, tahmin BENIGN, kontrolden geçti        → izin
UDP paketleri atılır (çıkarıcı doğrulaması: veri setinin yazarları UDP'yi kaydetmemiş, model hiç görmedi).

Kullanım (yerel bilgisayar; tshark kurulu olmalı):
  python src/demo/replay_dnn_edgeiiot.py --bundle <demo_bundle klasörü> --scenario WEB \\
      --pcap "<.../Temperature_and_Humidity.pcap>:Normal" --pcap "<.../XSS attacks.pcap>:XSS" \\
      --pcap "<.../Port Scanning attack.pcap>:Port_Scanning" --max-packets 20000 --events olaylar.jsonl
  --live            her ALARM/KONTROL kararını anında yazar
  --speed 50        kayıt zamanını 50 kat hızlandırarak oynatır (vermezsen beklemeden işler)
  --db              kararları, bağlantıları ve analiste gidenleri veritabanına yazar (.env'deki DATABASE_URL;
                    bkz. src/demo/db_dnn_edgeiiot.py). --db sqlite:///… ile adres verilebilir.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from extractor_dnn_edgeiiot import extract, model_matrix, protocol_of

DECISIONS = ["NORMAL", "SALDIRI", "KONTROL", "ALARM"]


# ---------------------------------------------------------------- skorlar (open_set_lofo.scores_from_members ile aynı)
def _entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(-1)


def tabm_scores(model, X):
    """X (n, 32) → olasılık (n, C), üye anlaşmazlığı, enerji. Deneylerdeki formülün aynısı:
    anlaşmazlık = H(ortalama P) − ortalama H(üye P); enerji = −logsumexp(ortalama logit)."""
    logits = model._logits(model._prep(X))                      # (n, k, C)
    z = logits - logits.max(-1, keepdims=True)
    Pm = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    P = Pm.mean(axis=1)
    disagree = _entropy(P) - _entropy(Pm).mean(axis=1)
    energy = -logsumexp(logits.mean(axis=1), axis=-1)
    return P, disagree, energy


# ---------------------------------------------------------------- senaryo
class Scenario:
    def __init__(self, bundle, family):
        import joblib
        d = os.path.join(bundle, family)
        if not os.path.isdir(d):
            raise SystemExit(f"Pakette '{family}' yok: {sorted(os.listdir(bundle))}")
        self.meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.model = joblib.load(os.path.join(d, "model.joblib"))
        self.known = self.meta["known_classes"]
        self.b = self.meta["benign_index"]
        self.tau_alarm, self.tau_guard = self.meta["tau_alarm"], self.meta["tau_guard"]
        self.iqr = self.meta["score_iqr"]

    def decide(self, Xc):
        if list(Xc.columns) != self.meta["features"]:
            raise SystemExit("Çıkarıcının sütunları paketteki modelin beklediği sırayla aynı değil")
        P, dis, en = tabm_scores(self.model, Xc.to_numpy(np.float32))
        pred = P.argmax(1)
        alarm = dis > self.tau_alarm
        benign = pred == self.b
        guard = ~alarm & benign & (en > self.tau_guard)
        decision = np.where(alarm, "ALARM", np.where(guard, "KONTROL", np.where(benign, "NORMAL", "SALDIRI")))
        return pd.DataFrame({"decision": decision, "pred_class": np.array(self.known)[pred],
                             "confidence": P.max(1), "disagree": dis, "energy": en,
                             # eşiğe uzaklık, kalibrasyon IQR'si biriminde (> 0 ise eşik aşıldı)
                             "alarm_margin": (dis - self.tau_alarm) / self.iqr["disagree"],
                             "guard_margin": (en - self.tau_guard) / self.iqr["energy"]})


def family_of(label):
    from common.taxonomy import edgeiiot_category
    try:
        return edgeiiot_category(label)
    except ValueError:
        return None


def load_sources(specs, max_packets, scen):
    """--pcap YOL:ETİKET → birleşik olay tablosu (göreli zamana göre sıralı)."""
    frames = []
    for spec in specs:
        path, _, label = spec.rpartition(":")
        if not path:
            path, label = spec, "?"
        t0 = time.perf_counter()
        meta, X = extract(path, max_packets)
        keep = (protocol_of(meta) != "udp").to_numpy()
        meta, X = meta[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        dec = scen.decide(model_matrix(X, scen.meta["condition"]))
        t = meta["frame.time_epoch"].to_numpy(float)
        df = pd.DataFrame({"t_rel": t - np.nanmin(t), "source": os.path.basename(path), "true_label": label,
                           "true_family": family_of(label), "frame": meta["frame.number"].astype(int),
                           "src": meta["ip.src"].str.split(",").str[0], "dst": meta["ip.dst"].str.split(",").str[0],
                           "proto": protocol_of(meta).to_numpy(),
                           "dport": X["tcp.dstport"].astype(int).to_numpy(),
                           "sport": pd.to_numeric(meta["tcp.srcport"].str.split(",").str[0], errors="coerce")
                                      .fillna(0).astype(int).to_numpy(),
                           "tcp_len": X["tcp.len"].astype(int).to_numpy()})
        df = pd.concat([df, dec], axis=1)
        frames.append(df)
        print(f"  {os.path.basename(path)} ({label}): {len(df)} paket (UDP atıldı: {int((~keep).sum())}), "
              f"{time.perf_counter() - t0:.1f} sn")
    ev = pd.concat(frames, ignore_index=True).sort_values("t_rel", kind="stable").reset_index(drop=True)
    return ev


def connection_ids(ev):
    """Bağlantı = protokol + iki uç (adres:port), yönden bağımsız. TCP dışı paketler (ARP, ICMP) kendi başına birimdir."""
    a = ev["src"].astype(str) + ":" + ev["sport"].astype(str)
    b = ev["dst"].astype(str) + ":" + ev["dport"].astype(str)
    lo, hi = np.where(a < b, a, b), np.where(a < b, b, a)
    cid = pd.Series(ev["proto"].astype(str).to_numpy() + "|" + lo + "|" + hi, index=ev.index)
    single = ev["proto"].to_numpy() != "tcp"
    cid[single] = "pkt|" + ev.loc[single, "source"].astype(str) + "|" + ev.loc[single, "frame"].astype(str)
    return cid


def kind_of(fam, unseen):
    return "normal" if fam == "BENIGN" else ("GÖRÜLMEMİŞ" if fam == unseen else "bilinen")


def summarize(ev, scen):
    unseen = scen.meta["unseen_family"]
    print(f"\nÖzet (senaryo: model {unseen} ailesini hiç görmedi; eşikler alarm {scen.tau_alarm:.4g}, "
          f"kontrol {scen.tau_guard:.4g}):")
    rows = []
    ev = ev.assign(conn=connection_ids(ev), flagged=ev["decision"] != "NORMAL")
    for (src, lab), g in ev.groupby(["source", "true_label"], sort=False):
        share = g["decision"].value_counts(normalize=True)
        kind = kind_of(g["true_family"].iloc[0], unseen)
        r = {"kaynak": src, "etiket": lab, "tür": kind, "paket": len(g),
             **{d: round(float(share.get(d, 0.0)) * 100, 2) for d in DECISIONS}}
        if kind == "normal":
            r["yorum"] = f"analiste giden normal trafik %{r['ALARM'] + r['KONTROL']:.2f}"
        else:
            caught = r["ALARM"] + r["KONTROL"] + r["SALDIRI"]
            r["yorum"] = (f"durdurulan %{caught:.2f}, sessiz geçen (NORMAL) %{r['NORMAL']:.2f}"
                          + ("" if kind == "bilinen" else f"; bilinmeyen diye işaretlenen %{r['ALARM'] + r['KONTROL']:.2f}"))
            top = g.loc[g["decision"] == "SALDIRI", "pred_class"].value_counts(normalize=True).head(2)
            if len(top):
                r["yorum"] += "; saldırı sanılan sınıf " + ", ".join(f"{k} %{v * 100:.0f}" for k, v in top.items())
        # paket düzeyinin ayrıntısı ve bağlantı düzeyi (raporlama; kararlar değişmez)
        pay = (g["tcp_len"] > 0).to_numpy()
        r["sessiz_icerikli_%"] = round(float(((g["decision"] == "NORMAL").to_numpy() & pay).sum() / max(pay.sum(), 1)) * 100, 2)
        r["sessiz_bos_%"] = round(float(((g["decision"] == "NORMAL").to_numpy() & ~pay).sum() / max((~pay).sum(), 1)) * 100, 2)
        conn = g.groupby("conn")["flagged"].any()
        r["baglanti"] = int(len(conn))
        r["isaretli_baglanti_%"] = round(float(conn.mean()) * 100, 2)
        # Bağlantı kuralı (29.10/30.10.2026, XSS/Port_Scanning/Temperature sonuçlarına bakılarak tasarlandı; başka PCAP'lerde
        # doğrulanacak): bağlantı işaretlenir ⇔ içerikli (tcp.len > 0) bir paketi NORMAL değil VEYA bir paketi SALDIRI.
        # Boş paketlere verilen ALARM/KONTROL tek başına işaretlemez (boş ACK/SYN/FIN normalin ikizi; bilgi taşımaz).
        rule = (g["flagged"] & (g["tcp_len"] > 0)) | (g["decision"] == "SALDIRI")
        cr = rule.groupby(g["conn"]).any()
        has_pay = (g["tcp_len"] > 0).groupby(g["conn"]).any()
        r["kural_baglanti_%"] = round(float(cr.mean()) * 100, 2)
        r["icerikli_baglanti"] = int(has_pay.sum())
        r["kural_icerikli_baglanti_%"] = round(float(cr[has_pay].mean()) * 100, 2) if has_pay.any() else None
        rows.append(r)
    tab = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.drop(columns="yorum").to_string(index=False))
    for r in rows:
        print(f"  {r['etiket']:>16s} ({r['tür']}): {r['yorum']}")
    print("\nAyrıntı (raporlama; kararlar aynı):")
    for r in rows:
        what = "analiste giden normal bağlantı" if r["tür"] == "normal" else "fark edilen (en az bir paketi işaretli) bağlantı"
        print(f"  {r['etiket']:>16s}: NORMAL denen paketler içerikli paketlerin %{r['sessiz_icerikli_%']:.2f}'i, "
              f"boş (tcp.len = 0) paketlerin %{r['sessiz_bos_%']:.2f}'i; {r['baglanti']} bağlantı, {what} %{r['isaretli_baglanti_%']:.2f}")
    print("\nBağlantı kuralı (içerikli paket işaretli VEYA paket SALDIRI; boş pakete verilen ALARM/KONTROL tek başına saymaz):")
    for r in rows:
        what = "analiste giden normal bağlantı" if r["tür"] == "normal" else "fark edilen bağlantı"
        print(f"  {r['etiket']:>16s}: {what} %{r['kural_baglanti_%']:.2f} (tüm {r['baglanti']} bağlantı); "
              f"içerik taşıyan {r['icerikli_baglanti']} bağlantıda %{r['kural_icerikli_baglanti_%']}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="demo_bundle.zip'in açıldığı klasör (içinde WEB/, BRUTEFORCE/ …)")
    p.add_argument("--scenario", default="WEB", help="Görülmemiş aile: WEB, BRUTEFORCE, RECON, DDOS, MALWARE")
    p.add_argument("--pcap", action="append", required=True, help="YOL:ETİKET (etiket veri setindeki Attack_type, ör. XSS, Normal)")
    p.add_argument("--max-packets", type=int, default=20000, help="PCAP başına en fazla paket (0 = tamamı)")
    p.add_argument("--speed", type=float, default=0, help="Kayıt zamanını bu kat hızlandırarak oynat (0 = beklemeden)")
    p.add_argument("--live", action="store_true", help="ALARM ve KONTROL kararlarını anında yaz")
    p.add_argument("--events", default=None, help="Bütün kararları JSONL olarak yaz (arayüz/veritabanı için)")
    p.add_argument("--db", nargs="?", const="", default=None,
                   help="Veritabanına yaz (adres verilmezse .env'deki DATABASE_URL)")
    p.add_argument("--note", default=None, help="Veritabanındaki oynatma kaydına not")
    args = p.parse_args()

    scen = Scenario(args.bundle, args.scenario)
    print(f"Senaryo {args.scenario}: model bilinen sınıflar {scen.known}; görülmemiş {scen.meta['unseen_fine_labels']}")
    ev = load_sources(args.pcap, args.max_packets, scen)

    if args.live or args.speed:
        start = time.perf_counter()
        for r in ev.itertuples(index=False):
            if args.speed:
                wait = r.t_rel / args.speed - (time.perf_counter() - start)
                if wait > 0:
                    time.sleep(wait)
            if args.live and r.decision in ("ALARM", "KONTROL"):
                why = (f"anlaşmazlık eşiğin {r.alarm_margin:+.1f} IQR üstünde" if r.decision == "ALARM"
                       else f"'normal' dedi ama enerji eşiğin {r.guard_margin:+.2f} IQR üstünde")
                print(f"[{r.t_rel:8.2f} sn] {r.decision:7s} {r.src} → {r.dst}:{r.dport}  ({r.true_label}) {why}")
    rows = summarize(ev, scen)
    if args.events:
        ev.to_json(args.events, orient="records", lines=True, force_ascii=False)
        print(f"Olaylar: {args.events} ({len(ev)} satır)")
    if args.db is not None:
        import db_dnn_edgeiiot as dbm
        url = dbm.database_url(args.db or None)
        db = dbm.DB(url)
        dbm.init(db)
        unseen = scen.meta["unseen_family"]
        kinds = {s: kind_of(f, unseen) for s, f in ev.groupby("source")["true_family"].first().items()}
        run_id, n_conn, n_alert = dbm.save_run(db, ev.assign(conn=connection_ids(ev)), scen, args.bundle,
                                               args.max_packets, kinds, args.note)
        db.close()
        print(f"Veritabanı ({dbm.safe_url(url)}): oynatma {run_id}; {len(ev)} paket, {n_conn} bağlantı, "
              f"analiste giden {n_alert}")
    return rows


if __name__ == "__main__":
    main()
