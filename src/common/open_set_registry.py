"""Görülmemiş saldırı (LOFO) deneyi için veri seti kaydı (ekip ortak aracı).

Her veri seti burada BİR kez tanımlanır; deney betikleri (open_set_lofo.py, cp_tune_dnn_edgeiiot.py)
veri setine özel hiçbir şey bilmez. Yeni bir veri seti eklemek için gereken her şey:

  loader(data_file, cap, seed) -> X, fine, category, stats
      X        : pandas DataFrame, float32, yalnızca model özellikleri
      fine     : np.array, ince etiketler (büyük harf)
      category : np.array, ortak kategoriler (common/taxonomy.py)
      stats    : dict, veri hazırlama sayıları (raporlama için)
  category_of(fine_label) -> ortak kategori      (alt türlerle eğitim için)
  families    : {aile adı: [kategoriler]}       bir aile birden fazla kategori kapsayabilir
                                                (ör. CICIoT2023'te FLOOD = DOS + DDOS)
  conditions  : {koşul adı: [çıkarılacak sütunlar]}  A tam, C ana koşul, E en sert
  benign      : zararsız kategorinin adı (sessiz kaçış ölçütü için)
  default_condition : ana koşul

Koşul sütunları veri yüklendikten SONRA çıkarılır; böylece her koşulda satırlar birebir aynıdır.
Kayıtta adı geçen ama veride olmayan bir sütun deneyi DURDURUR (yazım hatası sessizce geçmesin).
"""


def _edge_loader(data_file, cap, seed):
    from model_dnn_edgeiiot import load_data
    return load_data(data_file, cap, seed, drop_conflicts=True)


def _edge_category(label):
    from common.taxonomy import edgeiiot_category
    return edgeiiot_category(label)


# Kişi 1'in resmi protokolü (model_ml.py, 18.10.2026): Number her zaman atılır (veri toplama izi:
# floodlar 100, diğer sınıflar 10 paketlik pencereyle üretilmiş). model_ml.load_data bunu yalnızca
# drop_features ile yapar; komut satırındaki varsayılan burada açıkça tekrarlanır.
CICIOT_PROTOCOL_DROP = ("Number",)


def _ciciot_loader(data_file, cap, seed):
    # Kişi 1'in yükleyicisi: data_file = Merged*.csv dosyalarının bulunduğu klasör
    from model_ml import load_data
    X, fine, category, stats = load_data(data_file, cap, seed, drop_features=CICIOT_PROTOCOL_DROP,
                                         drop_conflicts=True)
    leaked = [c for c in CICIOT_PROTOCOL_DROP if c in X.columns]
    if leaked:  # yükleyici değişirse sessizce geri gelmesin
        raise SystemExit(f"CICIoT2023: protokol gereği atılması gereken sütun(lar) veride: {leaked}")
    return X, fine, category, stats


def _ciciot_category(label):
    from common.taxonomy import ciciot2023_category
    return ciciot2023_category(label)


_EDGE_PORTS = ["tcp.dstport", "udp.port"]
_EDGE_SEQ_ACK = ["tcp.seq", "tcp.ack"]
_EDGE_COUNTERS = ["mbtcp.trans_id", "icmp.transmit_timestamp"]  # ablation_dnn_edgeiiot.SUSPECT_IDS

DATASETS = {
    "dnn_edgeiiot": {
        "name": "Edge-IIoTset",
        "owner": "Kişi 3",
        "loader": _edge_loader,
        "category_of": _edge_category,
        "benign": "BENIGN",
        "families": {"DDOS": ["DDOS"], "RECON": ["RECON"], "WEB": ["WEB"],
                     "BRUTEFORCE": ["BRUTEFORCE"], "MALWARE": ["MALWARE"]},
        # ablation_dnn_edgeiiot.SCENARIOS ile aynı (A, C, E raporlanır; B, D ablation için)
        "conditions": {"A_tam": [],
                       "B_portsuz": _EDGE_PORTS,
                       "C_seq_ack_yok": _EDGE_SEQ_ACK,
                       "D_sayac_zaman_yok": _EDGE_COUNTERS,
                       "E_hepsi_cikarildi": _EDGE_PORTS + _EDGE_SEQ_ACK + _EDGE_COUNTERS},
        "default_condition": "C_seq_ack_yok",
        "notes": "SPOOFING yok (MITM satırları sütun kayması nedeniyle atıldı); Backdoor MALWARE'de "
                 "(Edge-IIoTset makalesinin gruplaması).",
    },
    "dataset_23": {
        "name": "CICIoT2023",
        "owner": "Kişi 1",
        "loader": _ciciot_loader,
        "category_of": _ciciot_category,
        "benign": "BENIGN",
        # DoS ve DDoS veride ayırt edilemiyor (çakışmaların %82'si, birleşince F1 1,0): tek aile
        "families": {"FLOOD": ["DOS", "DDOS"], "MIRAI": ["MIRAI"], "RECON": ["RECON"],
                     "SPOOFING": ["SPOOFING"], "WEB": ["WEB"], "BRUTEFORCE": ["BRUTEFORCE"]},
        # Kişi 1'in belirlediği kısayol koşulları (README, dataset_23-lofo dalı); sütun adlarını
        # Kişi 1 doğrular: yazım farkı varsa deney açık bir hata ile durur.
        "conditions": {"A_tam": [],
                       "C_ttl_yok": ["Time_To_Live"],
                       "E_hepsi_cikarildi": ["Time_To_Live", "IAT", "Rate", "Header_Length"]},
        "default_condition": "C_ttl_yok",
        "notes": "Backdoor_Malware WEB'de (CICIoT2023 resmi gruplaması).",
    },
}

# Veri setleri arasında doğrudan karşılaştırılabilen aileler (aynı tanım)
COMPARABLE = {"FLOOD~DDOS": {"dataset_23": "FLOOD", "dnn_edgeiiot": "DDOS"},
              "RECON": {"dataset_23": "RECON", "dnn_edgeiiot": "RECON"},
              "WEB": {"dataset_23": "WEB", "dnn_edgeiiot": "WEB"},
              "BRUTEFORCE": {"dataset_23": "BRUTEFORCE", "dnn_edgeiiot": "BRUTEFORCE"}}


def get(dataset_id):
    if dataset_id not in DATASETS:
        raise SystemExit(f"Bilinmeyen veri seti: {dataset_id} (kayıtlı: {list(DATASETS)})")
    return DATASETS[dataset_id]


def apply_condition(X, spec, condition):
    """Koşulun sütunlarını çıkarır; kayıtta olup veride olmayan sütun varsa durur."""
    if condition not in spec["conditions"]:
        raise SystemExit(f"{spec['name']} için koşul yok: {condition} (kayıtlı: {list(spec['conditions'])})")
    drop = list(spec["conditions"][condition])
    missing = [c for c in drop if c not in X.columns]
    if missing:
        raise SystemExit(f"{spec['name']} / {condition}: veride olmayan sütun(lar) {missing}. "
                         f"Mevcut sütunlar: {list(X.columns)}")
    return X.drop(columns=drop), drop
