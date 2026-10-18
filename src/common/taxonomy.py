"""Ekibin ortak saldırı taksonomisi.

Her veri seti kendi etiketlerini bu ortak kategorilere eşler. DSS (main.py)
yalnızca bu kategorileri ve ciddiyet seviyelerini bilir. Yeni bir kategori
gerekiyorsa buraya ekleyin ve PR açın; herkes aynı listeyi kullanmalı.
"""

# Ortak kategori -> ciddiyet (0 = zararsız, 3 = kritik)
SEVERITY = {
    "BENIGN": 0,
    "RECON": 1,
    "SPOOFING": 2,
    "BRUTEFORCE": 2,
    "WEB": 2,
    "MALWARE": 3,
    "DOS": 3,
    "DDOS": 3,
    "MIRAI": 3,
}

COMMON_CLASSES = list(SEVERITY)


def ciciot2023_category(label: str) -> str:
    """CICIoT2023 34 sınıflık etiketi 8 ortak kategoriye eşler (resmi gruplama)."""
    label = label.strip().upper()
    if label == "BENIGN":
        return "BENIGN"
    if label.startswith("DDOS-"):
        return "DDOS"
    if label.startswith("DOS-"):
        return "DOS"
    if label.startswith("MIRAI-"):
        return "MIRAI"
    if label.startswith("RECON-") or label == "VULNERABILITYSCAN":
        return "RECON"
    if label in {"DNS_SPOOFING", "MITM-ARPSPOOFING"}:
        return "SPOOFING"
    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTEFORCE"
    if label in {"BROWSERHIJACKING", "XSS", "UPLOADING_ATTACK", "SQLINJECTION",
                 "COMMANDINJECTION", "BACKDOOR_MALWARE"}:
        return "WEB"
    raise ValueError(f"Eşlenmemiş CICIoT2023 etiketi: {label}")


# Edge-IIoTset (dataset_26): 14 saldırı + Normal -> ortak kategoriler.
# Kaynak gruplama: Ferrag vd., IEEE Access 2022 (5 tehdit ailesi), ekip taksonomisine uyarlandı.
EDGEIIOT_MAP = {
    "NORMAL": "BENIGN",
    "DDOS_UDP": "DDOS", "DDOS_ICMP": "DDOS", "DDOS_TCP": "DDOS", "DDOS_HTTP": "DDOS",
    "PORT_SCANNING": "RECON", "FINGERPRINTING": "RECON", "VULNERABILITY_SCANNER": "RECON",
    "MITM": "SPOOFING",                       # ARP/DNS sahteciliği (CICIoT2023'teki MITM-ARPSPOOFING ile aynı)
    "SQL_INJECTION": "WEB", "XSS": "WEB", "UPLOADING": "WEB",
    "PASSWORD": "BRUTEFORCE",                 # parola kırma
    "BACKDOOR": "MALWARE", "RANSOMWARE": "MALWARE",
}


def edgeiiot_category(label: str) -> str:
    """Edge-IIoTset Attack_type etiketini ortak kategoriye eşler."""
    key = label.strip().upper()
    if key not in EDGEIIOT_MAP:
        raise ValueError(f"Eşlenmemiş Edge-IIoTset etiketi: {label}")
    return EDGEIIOT_MAP[key]
