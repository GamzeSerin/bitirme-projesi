"""Arayüz için SENTETİK örnek oynatma (yalnız deneme; model ve PCAP gerekmez). Kararlar rastgele üretilir, gerçek değildir.

Kullanım (depo kökünden):
  python src/demo/web/make_sample_dnn_edgeiiot.py sqlite:///outputs/dnn_edgeiiot/demo/ornek.db
  DATABASE_URL=sqlite:///outputs/dnn_edgeiiot/demo/ornek.db python src/demo/web/app_dnn_edgeiiot.py
Kaynaklar: 3 normal cihaz, Modbus (yeni cihaz), Password (bilinen; BRUTEFORCE), SQL_injection (görülmemiş; WEB).
"""
import sys, types, numpy as np, pandas as pd
sys.path.insert(0, "src/demo"); sys.path.insert(0, "src/demo/web"); sys.path.insert(0, "src")
import db_dnn_edgeiiot as dbm
from replay_dnn_edgeiiot import connection_ids
rng = np.random.default_rng(0)
known = ["BENIGN", "DDOS", "RECON", "BRUTEFORCE", "MALWARE"]
def src(name, label, fam, n, dur, attack=None, newdev=False):
    t = np.sort(rng.uniform(0, dur, n))
    ip = {"Normal": f"192.168.0.{rng.integers(10,99)}"}.get(label, "10.0.0.66")
    dec = np.full(n, "NORMAL", object); pred = np.full(n, "BENIGN", object)
    am = rng.normal(-2, 0.7, n); gm = rng.normal(-1.5, 0.8, n)
    if attack == "known": dec[rng.random(n) < .85] = "SALDIRI"; pred[dec == "SALDIRI"] = "BRUTEFORCE"
    if attack == "unseen":
        r = rng.random(n); dec[r < .45] = "ALARM"; dec[(r >= .45) & (r < .6)] = "KONTROL"; am[dec == "ALARM"] = rng.uniform(0.1, 3, (dec == "ALARM").sum())
        pred[r > .8] = "RECON"; dec[r > .8] = "SALDIRI"
    if newdev:
        r = rng.random(n); dec[r < .12] = "KONTROL"; gm[dec == "KONTROL"] = rng.uniform(0.1, 2, (dec == "KONTROL").sum())
    if not attack and not newdev: r = rng.random(n); dec[r < .004] = "KONTROL"
    sport = rng.integers(40000, 41000, n) if attack else rng.integers(50000, 50030, n)
    return pd.DataFrame({"t_rel": t, "source": name, "true_label": label, "true_family": fam, "frame": np.arange(1, n+1),
        "src": ip, "dst": "192.168.0.128", "proto": "tcp", "dport": 502 if newdev else (80 if attack else 1883), "sport": sport,
        "tcp_len": np.where(rng.random(n) < .5, rng.integers(1, 300, n), 0), "decision": dec, "pred_class": pred,
        "confidence": rng.uniform(.5, 1, n), "disagree": rng.uniform(0, .1, n), "energy": rng.normal(-8, 2, n),
        "alarm_margin": am, "guard_margin": gm})
ev = pd.concat([src("Temperature_and_Humidity.pcap", "Normal", "BENIGN", 900, 150),
                src("Water_Level.pcap", "Normal", "BENIGN", 700, 150),
                src("Heart_Rate.pcap", "Normal", "BENIGN", 500, 150),
                src("Modbus.pcap", "Normal", "BENIGN", 400, 40, newdev=True),
                src("Password attacks.pcap", "Password", "BRUTEFORCE", 6000, 40, attack="known"),
                src("SQL_injection_attack.pcap", "SQL_injection", "WEB", 800, 40, attack="unseen")]).sort_values("t_rel").reset_index(drop=True)
scen = types.SimpleNamespace(meta={"unseen_family": "WEB"}, known=known, tau_alarm=0.05, tau_guard=-6.0)
kinds = {"Temperature_and_Humidity.pcap": "normal", "Water_Level.pcap": "normal", "Heart_Rate.pcap": "normal", "Modbus.pcap": "normal",
         "Password attacks.pcap": "bilinen", "SQL_injection_attack.pcap": "GÖRÜLMEMİŞ"}
db = dbm.DB(sys.argv[1]); dbm.init(db)
print(dbm.save_run(db, ev.assign(conn=connection_ids(ev)), scen, "/tmp/bundle", 20000, kinds, "sentetik arayüz testi"))
